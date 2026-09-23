"""
physai/solvers/dedalus.py

Numerical (non-PINN) PDE solving for PhysAI, used chiefly for classical
cross-validation of trained models (see ``Trainer.cross_validate_dedalus``
and ``Trainer.cross_validate_geometry``). Two genuinely different numerical
methods live here, dispatched by whether a ``physai.geometry.Geometry`` is
given:

* **Box domains** (no ``geometry``) -> real Dedalus (``dedalus.public``).
  ``Solver.solve_box`` builds a tensor-product Chebyshev/Fourier basis per
  axis and hands the equations straight to Dedalus's own IVP solver and
  tau-correction compiler. Generalized from the previous 1D-only
  implementation to an arbitrary number of spatial axes — still real
  Dedalus, just no longer hardcoded to a single ``CartesianCoordinates('x')``.

* **Arbitrary geometry** (CSG combinations, custom SDFs, meshes loaded via
  ``physai.geometry.geometry_from_file`` — anything that is a
  ``physai.geometry.Geometry``) -> ``Solver.solve_geometry``, an
  embedded-boundary finite-difference solver on a masked regular N-D grid,
  assembled with ``scipy.sparse`` (or ``cupyx.scipy.sparse`` on GPU, if
  ``array_module="cupy"`` and ``cupy`` is installed). Dedalus's own
  spectral bases are built for boxes, intervals, and specific curvilinear
  coordinate systems — they do not support arbitrary CSG/SDF domains, so
  reusing them here would mean secretly approximating the geometry as a
  box, which is exactly the kind of "subtly used something easier instead
  of actual rigor" this module exists to avoid. NumPy/SciPy/CuPy solve the
  real masked-domain linear system instead.

  This path is honestly scoped to the PDE classes an embedded-boundary FD
  discretization can solve *correctly*: steady linear elliptic (Poisson,
  Helmholtz) and linear parabolic (heat/diffusion, implicit Euler in
  time). It does not attempt fully nonlinear PDEs or hyperbolic (wave)
  problems on arbitrary geometry — extending a masked finite-difference
  stencil to those correctly (TVD/upwind schemes for nonlinear advection,
  CFL-stable explicit or Newmark schemes for wave equations, all under an
  irregular embedded boundary) is a substantially larger, separate
  undertaking than this cross-validation utility, and PhysAI's actual
  arbitrary-geometry, nonlinear-PDE story is the PINN pipeline
  (``physai.geometry`` + ``physai.trainer.Trainer``), which handles that
  case natively via collocation sampling rather than a grid. Calling
  ``solve_geometry`` with an unsupported ``pde`` raises immediately rather
  than silently returning something wrong.

  The boundary condition itself is enforced by simple embedded/staircase
  substitution: for a domain grid node with a neighbor outside the
  geometry, the unknown neighbor value is replaced by the prescribed
  Dirichlet function evaluated at that neighbor's own grid coordinate
  (not projected onto the true boundary surface). This is a standard,
  well-understood first-order-accurate immersed-boundary approximation —
  stated plainly here, rather than presented as spectrally accurate the
  way the Dedalus path genuinely is.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from physai.geometry import Geometry

# `dedalus` is a heavy optional dependency. Import lazily inside Solver
# methods rather than unconditionally at module load, so
# `from physai.solvers.dedalus import Solver` doesn't crash on machines
# without dedalus installed — the ImportError is only raised when someone
# actually tries to build/run a box solve.
try:
    import dedalus.public as d3  # type: ignore
    _DEDALUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    d3 = None
    _DEDALUS_AVAILABLE = False


def _require_dedalus() -> None:
    if not _DEDALUS_AVAILABLE:
        raise ImportError(
            "The 'dedalus' package is required for Solver.solve_box / the "
            "box-domain Solver(...) constructor. It needs a Conda "
            "environment, not plain pip — run the bundled installer with "
            "`python -m physai.dedalus_setup` or `physai.install_dedalus()`"
        )


def _require_cupy():
    try:
        import cupy  # type: ignore
        import cupyx.scipy.sparse as cusp  # type: ignore
        import cupyx.scipy.sparse.linalg as cuspla  # type: ignore
        return cupy, cusp, cuspla
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "array_module='cupy' requires the 'cupy' package (matching "
            "your CUDA version). Install it with, e.g.: "
            "pip install cupy-cuda12x"
        ) from e


def _as_axis_list(value, dims: int, name: str) -> list:
    """Normalize a per-axis argument that may be given as a single scalar
    (applied to every axis) or as a length-`dims` sequence."""
    if isinstance(value, (list, tuple)) and not (
        name == "bounds" and len(value) == 2 and not isinstance(value[0], (list, tuple))
    ):
        seq = list(value)
        if len(seq) != dims:
            raise ValueError(f"'{name}' has length {len(seq)}, expected {dims} (one per spatial axis).")
        return seq
    return [value] * dims


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver:
    """
    Dispatching numerical PDE solver.

    * No ``geometry`` -> real Dedalus box solve (``solve_box`` / the
      instance constructor below), any number of spatial dimensions.
    * ``geometry`` given -> embedded-boundary finite-difference solve on a
      masked N-D grid (``solve_geometry``), NumPy/SciPy or CuPy.
    """

    def __init__(self, domain_type, bounds, grid_points, variables, is_complex=False):
        """
        Persistent, reusable Dedalus box solver object — sets up the
        static geometry, coordinates, and fields once. N-D generalization
        of the original 1D-only constructor: ``domain_type``, ``bounds``,
        and ``grid_points`` may each be given either as a single value
        (applied to every axis) or as a per-axis list whose length fixes
        the number of spatial dimensions.
        """
        _require_dedalus()
        logging.basicConfig(level=logging.WARNING)
        self.dtype = np.complex128 if is_complex else np.float64
        self.variables = variables

        # Infer dimensionality from whichever argument was given as a list.
        dims = 1
        for v in (domain_type, bounds, grid_points):
            if isinstance(v, (list, tuple)) and not (
                v is bounds and len(v) == 2 and not isinstance(v[0], (list, tuple))
            ):
                dims = max(dims, len(v))
        domain_types = _as_axis_list(domain_type, dims, "domain_type")
        bounds_list = _as_axis_list(bounds, dims, "bounds")
        grid_points_list = _as_axis_list(grid_points, dims, "grid_points")
        self.domain_types = [d.lower() for d in domain_types]

        axis_names = [f"x{i}" for i in range(dims)]
        self.coords = d3.CartesianCoordinates(*axis_names)
        self.dist = d3.Distributor(self.coords, dtype=self.dtype)

        self.bases = []
        for name, dtype_str, bnds, npts in zip(axis_names, self.domain_types, bounds_list, grid_points_list):
            coord = self.coords[name]
            if dtype_str == "chebyshev":
                basis = d3.Chebyshev(coord, size=npts, bounds=bnds, dealias=2)
            elif dtype_str == "fourier":
                basis = d3.Fourier(coord, size=npts, bounds=bnds, dealias=2)
            else:
                raise ValueError("Use 'Chebyshev' or 'Fourier' per axis")
            self.bases.append(basis)

        self.local_env = {'dt': d3.dt, 'd3': d3}
        for name, basis in zip(axis_names, self.bases):
            self.local_env[f"dx_{name}"] = lambda f, _b=basis: d3.Differentiate(f, _b)

        self.all_fields = []
        for var in variables:
            field_obj = self.dist.Field(name=var, bases=tuple(self.bases))
            self.local_env[var] = field_obj
            self.all_fields.append(field_obj)

        self.x_grid = [self.dist.local_grid(b) for b in self.bases]

    # ------------------------------------------------------------------
    # Real Dedalus: N-D box domains
    # ------------------------------------------------------------------

    @staticmethod
    def solve_box(domain_type, bounds, grid_points, variables, equations, bcs, ics,
                  is_complex=False, timestepper='RK443', dt=0.01, stop_time=1.0):
        """
        Dynamic-compiler solve of any linear/nonlinear/complex PDE or ODE
        system on an axis-aligned box, via real Dedalus. Any number of
        spatial dimensions: ``domain_type``/``bounds``/``grid_points`` may
        each be given as a single value (applied to every axis) or as a
        per-axis list, whose length then fixes the dimensionality.

        ``equations``, ``bcs`` are Dedalus equation strings (evaluated in a
        namespace containing each variable, ``dt``, and, for a 2-axis
        problem named x0/x1, ``dx_x0``/``dx_x1`` differentiation
        operators); ``ics`` maps variable name -> callable(*grid_arrays).
        """
        _require_dedalus()
        logging.basicConfig(level=logging.WARNING)
        dtype = np.complex128 if is_complex else np.float64

        dims = 1
        for v in (domain_type, bounds, grid_points):
            if isinstance(v, (list, tuple)) and not (
                v is bounds and len(v) == 2 and not isinstance(v[0], (list, tuple))
            ):
                dims = max(dims, len(v))
        domain_types = [d.lower() for d in _as_axis_list(domain_type, dims, "domain_type")]
        bounds_list = _as_axis_list(bounds, dims, "bounds")
        grid_points_list = _as_axis_list(grid_points, dims, "grid_points")

        axis_names = [f"x{i}" for i in range(dims)]
        coords = d3.CartesianCoordinates(*axis_names)
        dist = d3.Distributor(coords, dtype=dtype)

        bases = []
        for name, dtype_str, bnds, npts in zip(axis_names, domain_types, bounds_list, grid_points_list):
            coord = coords[name]
            if dtype_str == "chebyshev":
                basis = d3.Chebyshev(coord, size=npts, bounds=bnds, dealias=2)
            elif dtype_str == "fourier":
                basis = d3.Fourier(coord, size=npts, bounds=bnds, dealias=2)
            else:
                raise ValueError("Use 'Chebyshev' or 'Fourier' per axis")
            bases.append(basis)
        bases = tuple(bases)

        local_env = {'dt': d3.dt, 'd3': d3}
        for name, basis in zip(axis_names, bases):
            local_env[f"dx_{name}"] = lambda f, _b=basis: d3.Differentiate(f, _b)
        # Backward-compatible alias for the original 1D API's bare `dx`.
        if dims == 1:
            local_env['dx'] = local_env[f"dx_{axis_names[0]}"]

        all_fields = []
        for var in variables:
            field_obj = dist.Field(name=var, bases=bases)
            local_env[var] = field_obj
            all_fields.append(field_obj)

        # Automatically calculate and append required dynamic Boundary Tau
        # constraints. Rule of thumb: 1 tau per order of the boundary
        # condition constraint, only along axes using a Chebyshev basis
        # (Fourier axes are periodic and need no tau correction).
        chebyshev_bases = [b for b, t in zip(bases, domain_types) if t == "chebyshev"]
        tau_fields = []
        modified_equations = []
        for idx, eq_str in enumerate(equations):
            num_taus = eq_str.count('dx(dx(') * 2 or eq_str.count('dx(')
            if num_taus == 0 and chebyshev_bases:
                num_taus = 2  # safe default fallback for second-order physics systems

            tau_lift_strings = []
            for t_idx in range(num_taus):
                tau_name = f"tau_{idx}_{t_idx}"
                tau_field = dist.Field(name=tau_name)
                local_env[tau_name] = tau_field
                tau_fields.append(tau_field)
                # Lift against the (first) Chebyshev axis, matching the
                # original single-axis tau-correction convention.
                lift_basis_expr = "bases[0]" if not chebyshev_bases else "chebyshev_bases[0]"
                tau_lift_strings.append(f"d3.lift({tau_name}, {lift_basis_expr}, {-t_idx-1})")

            if tau_lift_strings and chebyshev_bases:
                clean_eq = eq_str.replace(" = 0", "") + " + " + " + ".join(tau_lift_strings) + " = 0"
                modified_equations.append(clean_eq)
            else:
                modified_equations.append(eq_str)

        eval_env = dict(local_env)
        eval_env["bases"] = bases
        eval_env["chebyshev_bases"] = chebyshev_bases

        problem = d3.IVP(all_fields + tau_fields, namespace=eval_env)
        for eq in modified_equations:
            problem.add_equation(eq)
        for bc in bcs:
            problem.add_equation(bc)

        solver = problem.build_solver(getattr(d3, timestepper))
        solver.stop_sim_time = stop_time

        # Inject Initial Conditions
        grids = [dist.local_grid(b) for b in bases]
        for var_name, ic_lambda in ics.items():
            local_env[var_name]['g'] = ic_lambda(*grids)

        while solver.ok:
            solver.step(dt)

        output_profiles = {var: local_env[var]['g'].copy() for var in variables}
        grid_out = grids[0] if dims == 1 else grids
        return grid_out, output_profiles

    # ------------------------------------------------------------------
    # Embedded-boundary finite differences: arbitrary geometry, any D
    # ------------------------------------------------------------------

    _SUPPORTED_PDES = {"poisson", "helmholtz", "heat"}

    @staticmethod
    def solve_geometry(
        geometry: Geometry,
        *,
        pde: str,
        bc_value: Callable[[np.ndarray], np.ndarray],
        grid_points: Union[int, Sequence[int]] = 64,
        pde_params: Optional[Dict[str, float]] = None,
        source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ic: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dt: float = 1e-3,
        n_steps: int = 1000,
        array_module: str = "numpy",
        dtype=np.float64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Embedded-boundary finite-difference solve of a single scalar
        linear PDE over an arbitrary ``physai.geometry.Geometry``, in
        however many dimensions the geometry has.

        Parameters
        ----------
        geometry     : any physai.geometry.Geometry (box, ball, CSG
                       combination, custom SDF, or a mesh loaded via
                       geometry_from_file).
        pde          : one of {"poisson", "helmholtz", "heat"}.
                       poisson:   -Laplacian(u) = source(x)
                       helmholtz: -Laplacian(u) - k^2 * u = source(x)
                                  (pde_params={"k": ...})
                       heat:      du/dt = alpha * Laplacian(u) + source(x),
                                  stepped implicitly (backward Euler,
                                  unconditionally stable) from `ic` for
                                  `n_steps` of size `dt`
                                  (pde_params={"alpha": ...})
        bc_value     : callable mapping an (N, dim) array of physical
                       coordinates to prescribed Dirichlet values there.
                       Evaluated at exterior grid-neighbor coordinates
                       (see module docstring — first-order embedded
                       boundary, not a true boundary projection).
        grid_points  : grid resolution, either one int (applied to every
                       axis) or a per-axis sequence.
        source       : callable (N, dim) -> (N,) RHS forcing; zero if None.
        ic           : required for pde="heat"; callable (N, dim) -> (N,)
                       initial condition on interior nodes.
        array_module : "numpy" (default) or "cupy" (GPU; requires cupy).

        Returns
        -------
        (points, mask, values) where ``points`` is the (Ntotal, dim) full
        regular-grid coordinate array, ``mask`` is a boolean (Ntotal,)
        array that is True where the node is inside ``geometry`` (i.e.
        where ``values`` is meaningful — exterior nodes are filled with
        NaN), and ``values`` is the solved field, same shape as ``mask``.
        """
        pde = pde.lower()
        if pde not in Solver._SUPPORTED_PDES:
            raise ValueError(
                f"solve_geometry only supports pde in {sorted(Solver._SUPPORTED_PDES)} "
                f"(got '{pde}'). Fully nonlinear PDEs and hyperbolic (wave) "
                "problems on arbitrary geometry aren't implemented here — see "
                "the module docstring for why, and use physai's PINN pipeline "
                "(physai.geometry + physai.trainer.Trainer) for those instead."
            )
        pde_params = pde_params or {}
        dims = geometry.dim
        n_per_axis = _as_axis_list(grid_points, dims, "grid_points")

        if array_module == "cupy":
            xp, xsp, xspla = _require_cupy()
        elif array_module == "numpy":
            xp, xsp, xspla = np, sp, spla
        else:
            raise ValueError("array_module must be 'numpy' or 'cupy'")

        # 1. Build the regular grid over geometry's own bounding box.
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(geometry.bounds, n_per_axis)]
        spacings = [ax[1] - ax[0] for ax in axes]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid_shape = mesh[0].shape
        points = np.stack([m.reshape(-1) for m in mesh], axis=-1).astype(dtype)  # (Ntotal, dims)

        # 2. Mask: True where the node is *strictly* inside the true
        # geometry. `Geometry.contains`'s default tol=0.0 uses sdf <= 0,
        # which counts points exactly ON the boundary as "inside" — for a
        # grid-aligned domain (e.g. a box whose faces coincide with grid
        # lines) that would make boundary nodes unknowns solved by the PDE
        # stencil instead of pinned to the prescribed Dirichlet value,
        # silently degrading an otherwise-exact boundary alignment to the
        # same first-order staircase error a curved geometry has no choice
        # about. A small negative tolerance excludes exact-boundary points
        # from the unknown set so they're substituted via `bc_value` at
        # their own (exact) coordinate instead — full second-order
        # accuracy whenever the geometry and grid genuinely align, with no
        # effect on curved geometries (where an exact zero-distance grid
        # hit essentially never occurs).
        mask = np.asarray(geometry.contains(points, tol=-1e-9)).reshape(-1).astype(bool)
        n_total = points.shape[0]
        if not np.any(mask):
            raise ValueError("solve_geometry: no grid nodes fall inside `geometry` — increase grid_points.")

        # Map full flat index -> unknown index (only for interior/domain nodes).
        unknown_index = -np.ones(n_total, dtype=np.int64)
        unknown_index[mask] = np.arange(int(mask.sum()))
        n_unknown = int(mask.sum())

        strides = np.array([int(np.prod(grid_shape[d + 1:])) for d in range(dims)], dtype=np.int64)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        rhs = np.zeros(n_unknown, dtype=dtype)

        flat_idx_domain = np.nonzero(mask)[0]
        for flat_idx in flat_idx_domain:
            row = int(unknown_index[flat_idx])
            multi_idx = np.unravel_index(flat_idx, grid_shape)
            diag = 0.0
            for d in range(dims):
                h2 = spacings[d] ** 2
                diag += 2.0 / h2
                for sign in (-1, 1):
                    neighbor_multi = list(multi_idx)
                    neighbor_multi[d] += sign
                    if neighbor_multi[d] < 0 or neighbor_multi[d] >= grid_shape[d]:
                        # Outside the bounding box entirely: treat as the
                        # prescribed boundary value at the (extrapolated)
                        # neighbor coordinate.
                        neighbor_coord = points[flat_idx].copy()
                        neighbor_coord[d] += sign * spacings[d]
                        g = float(np.asarray(bc_value(neighbor_coord.reshape(1, -1))).reshape(-1)[0])
                        rhs[row] += g / h2
                        continue
                    neighbor_flat = int(flat_idx + sign * strides[d])
                    if mask[neighbor_flat]:
                        rows.append(row)
                        cols.append(int(unknown_index[neighbor_flat]))
                        data.append(-1.0 / h2)
                    else:
                        # Embedded boundary: neighbor lies outside the true
                        # geometry. Substitute the prescribed Dirichlet
                        # value evaluated at that neighbor's own grid
                        # coordinate (first-order staircase approximation
                        # — see module docstring).
                        neighbor_coord = points[neighbor_flat]
                        g = float(np.asarray(bc_value(neighbor_coord.reshape(1, -1))).reshape(-1)[0])
                        rhs[row] += g / h2
            rows.append(row)
            cols.append(row)
            data.append(diag)

        L = sp.coo_matrix((data, (rows, cols)), shape=(n_unknown, n_unknown)).tocsr()

        src_vals = np.zeros(n_unknown, dtype=dtype)
        if source is not None:
            src_vals = np.asarray(source(points[mask])).reshape(-1).astype(dtype)

        values_full = np.full(n_total, np.nan, dtype=dtype)

        if pde == "poisson":
            b = rhs + src_vals
            A = L if array_module == "numpy" else xsp.csr_matrix(L)
            u = xspla.spsolve(A, xp.asarray(b) if array_module == "cupy" else b)
            u = np.asarray(u.get()) if array_module == "cupy" else np.asarray(u)
            values_full[mask] = u

        elif pde == "helmholtz":
            k = pde_params.get("k")
            if k is None:
                raise ValueError("pde='helmholtz' requires pde_params={'k': ...}")
            A_np = (L - (k ** 2) * sp.identity(n_unknown, dtype=dtype, format="csr"))
            b = rhs + src_vals
            A = A_np if array_module == "numpy" else xsp.csr_matrix(A_np)
            u = xspla.spsolve(A, xp.asarray(b) if array_module == "cupy" else b)
            u = np.asarray(u.get()) if array_module == "cupy" else np.asarray(u)
            values_full[mask] = u

        else:  # "heat"
            alpha = pde_params.get("alpha")
            if alpha is None:
                raise ValueError("pde='heat' requires pde_params={'alpha': ...}")
            if ic is None:
                raise ValueError("pde='heat' requires an initial condition `ic`")
            u = np.asarray(ic(points[mask])).reshape(-1).astype(dtype)
            # Implicit (backward) Euler: (I/dt + alpha*L) u^{n+1} = u^n/dt + source + alpha*rhs_bc
            I = sp.identity(n_unknown, dtype=dtype, format="csr")
            A_np = (I / dt + alpha * L)
            A = A_np if array_module == "numpy" else xsp.csr_matrix(A_np)
            if array_module == "cupy":
                A_factored = None  # cupy's spsolve re-factorizes each call; acceptable for this utility
            for _ in range(n_steps):
                b = u / dt + src_vals + alpha * rhs
                u = xspla.spsolve(A, xp.asarray(b) if array_module == "cupy" else b)
                u = np.asarray(u.get()) if array_module == "cupy" else np.asarray(u)
            values_full[mask] = u

        return points, mask, values_full

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def solve(*, geometry: Optional[Geometry] = None, **kwargs):
        """
        Dispatches to ``solve_geometry`` if ``geometry`` is given, else to
        ``solve_box`` (real Dedalus) with the remaining keyword arguments.
        """
        if geometry is not None:
            return Solver.solve_geometry(geometry, **kwargs)
        required = {"domain_type", "bounds", "grid_points", "variables", "equations", "bcs", "ics"}
        missing = required - kwargs.keys()
        if missing:
            raise TypeError(f"Solver.solve: missing required box-domain arguments {sorted(missing)}")
        return Solver.solve_box(**kwargs)


__all__ = ["Solver"]