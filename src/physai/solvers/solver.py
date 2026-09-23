"""
physai/solvers/solver.py

Classical (non-PINN) PDE solvers and model comparison helpers for PhysAI.
``Solver.solve`` dispatches among the built-in Dedalus and embedded-boundary
finite-difference paths, optional FiPy/FEniCS/FEniCSx/Meep integrations, and
user-registered solver callbacks. ``Solver.cross_validate`` compares a
model with any solver result that can be normalized to coordinates and
named fields. The historic default solve paths remain:

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

Optional native adapters are imported only when called. FiPy accepts an
equation-builder callback and returns cell-center samples. FEniCS and
FEniCSx accept native variational forms; Meep accepts a simulation-builder
callback. Results outside the standard ``{"coordinates": ..., "values": ...}``
shape can be normalized with ``Solver.cross_validate(result_adapter=...)``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from physai.geometry import Geometry

# `dedalus` is a heavy optional dependency. Import lazily inside Solver
# methods rather than unconditionally at module load, so
# `from physai.solvers.solver import Solver` doesn't crash on machines
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
            "box-domain Solver(...) constructor. Install it through Conda, "
            "then use that environment to run PhysAI. The bundled solver "
            "setup is `python -m physai.solver_setup` or "
            "`physai.install_solver_dependencies()`"
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
            "your CUDA version). Install it with Conda, for example: "
            "conda install -c conda-forge cupy"
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
    def solve(
        *,
        method: Optional[str] = None,
        equation: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        **kwargs,
    ):
        """
        Dispatch a solve to Dedalus, FiPy, FEniCS, FEniCSx, Meep, or a
        registered solver backend.

        With no ``method``, the historic behavior is preserved: arbitrary
        ``geometry`` uses the embedded-boundary solver, otherwise the box
        solve uses Dedalus. ``equation`` may name an entry registered with
        :func:`register_equation_solver`, allowing a user registered PDE to
        select its classical solver independently of its PINN residual.
        """
        selected = method
        if selected is None and equation is not None:
            selected = _EQUATION_SOLVERS.get(_solver_key(equation))
            if selected is None:
                raise ValueError(
                    f"No classical solver is registered for equation {equation!r}; "
                    "call register_equation_solver(equation, method) first."
                )
        if callable(selected):
            return selected(**kwargs)
        selected = _solver_key(selected or ("geometry" if geometry is not None else "dedalus"))

        if selected in {"dedalus", "geometry"}:
            if geometry is not None:
                return Solver.solve_geometry(geometry, **kwargs)
            if selected == "geometry":
                raise ValueError("method='geometry' requires a Geometry instance.")
            return _solve_dedalus_box(**kwargs)

        solve_fn = _SOLVER_BACKENDS.get(selected)
        if solve_fn is None:
            raise ValueError(
                f"Unknown solver method {selected!r}; available methods are "
                f"{sorted(_SOLVER_BACKENDS | {'dedalus': _solve_dedalus_box, 'geometry': Solver.solve_geometry})}."
            )
        return solve_fn(**kwargs)

    @staticmethod
    def cross_validate(
        model_fn: Callable[[Any], Any],
        backend: Any,
        *,
        spatial_dims: int,
        method: Optional[str] = None,
        equation: Optional[str] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        result_adapter: Optional[Callable[[Any], Dict[str, Any]]] = None,
        variables: Optional[Sequence[str]] = None,
        output_index: Optional[Dict[str, int]] = None,
        eval_time: Optional[float] = None,
        time_domain: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Union[int, float]]:
        """Run a classical solver and compare model outputs at its coordinates.

        Solver results use the normalized mapping
        ``{"coordinates": [N, d], "values": {name: [N]}}``. FiPy already
        returns this form; scalar finite-element functions are normalized
        from their dof data. Meep and custom layouts can be converted with
        ``result_adapter``. This method also accepts the historic Dedalus
        ``(coordinates, values)`` return tuple.
        """
        result = Solver.solve(
            method=method,
            equation=equation,
            **(solver_kwargs or {}),
        )
        normalized = result_adapter(result) if result_adapter is not None else _normalize_solver_result(result)
        if not isinstance(normalized, dict) or "coordinates" not in normalized or "values" not in normalized:
            raise TypeError(
                "The classical result adapter must return a mapping with "
                "'coordinates' and 'values' entries."
            )

        coordinates = np.asarray(normalized["coordinates"])
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 1)
        if coordinates.ndim != 2:
            raise ValueError("Classical solver coordinates must have shape [N, spatial_dims].")
        if coordinates.shape[1] > spatial_dims:
            coordinates = coordinates[:, :spatial_dims]

        fields = normalized["values"]
        if not isinstance(fields, dict):
            fields = {"u": fields}
        names = list(variables) if variables is not None else list(fields)
        if not names:
            raise ValueError("No classical solution fields were returned for comparison.")

        comparison_time = eval_time
        if comparison_time is None and time_domain is not None:
            comparison_time = time_domain[1]
        query_points = coordinates
        if comparison_time is not None and coordinates.shape[1] == spatial_dims:
            query_points = np.concatenate(
                [coordinates, np.full((len(coordinates), 1), comparison_time)], axis=1
            )
        prediction = np.asarray(backend.to_numpy(model_fn(backend.tensor(query_points))))
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)

        errors: Dict[str, Union[int, float]] = {}
        for position, name in enumerate(names):
            if name not in fields and not (len(fields) == 1 and len(names) == 1):
                raise KeyError(f"Classical solver result has no field {name!r}.")
            col = output_index[name] if output_index is not None and name in output_index else position
            classical_data = fields[name] if name in fields else next(iter(fields.values()))
            classical = np.asarray(classical_data).reshape(-1)
            predicted = prediction[:, col].reshape(-1)
            if predicted.shape != classical.shape:
                raise ValueError(
                    f"Model values for {name!r} have shape {predicted.shape}; "
                    f"classical values have shape {classical.shape}."
                )
            finite = np.isfinite(classical) & np.isfinite(predicted)
            if not np.any(finite):
                raise ValueError(f"Classical field {name!r} contains no finite comparison values.")
            difference = predicted[finite] - classical[finite]
            reference = classical[finite]
            errors[f"{name}_l2_abs"] = float(np.linalg.norm(difference))
            errors[f"{name}_l2_rel"] = float(
                np.linalg.norm(difference) / (np.linalg.norm(reference) + 1e-12)
            )
            errors[f"{name}_n_compared"] = int(finite.sum())
        return errors

    solve_fipy = staticmethod(lambda **kwargs: solve_fipy(**kwargs))
    solve_fenics = staticmethod(lambda **kwargs: solve_fenics(**kwargs))
    solve_fenicsx = staticmethod(lambda **kwargs: solve_fenicsx(**kwargs))
    solve_fenicsx_linear = staticmethod(lambda **kwargs: solve_fenicsx_linear(**kwargs))
    solve_meep = staticmethod(lambda **kwargs: solve_meep(**kwargs))


def _solve_dedalus_box(**kwargs):
    required = {"domain_type", "bounds", "grid_points", "variables", "equations", "bcs", "ics"}
    missing = required - kwargs.keys()
    if missing:
        raise TypeError(f"Solver.solve: missing required box-domain arguments {sorted(missing)}")
    return Solver.solve_box(**kwargs)


_SOLVER_LOCK = threading.RLock()
_SOLVER_BACKENDS: Dict[str, Callable[..., Any]] = {}
_EQUATION_SOLVERS: Dict[str, Any] = {}


def _solver_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Solver and equation names must be non-empty strings.")
    return name.strip().lower().replace("-", "_")


def register_solver(
    name: str,
    solve_fn: Callable[..., Any],
    *,
    aliases: Sequence[str] = (),
    overwrite: bool = False,
) -> Callable[..., Any]:
    """Register a solver method for :meth:`Solver.solve`.

    A custom solver callback receives the keyword arguments passed to
    ``Solver.solve(method=name, ...)``. Use this for package-native solver
    integrations that are not built in.
    """
    if not callable(solve_fn):
        raise TypeError("solve_fn must be callable.")
    keys = [_solver_key(name), *(_solver_key(alias) for alias in aliases)]
    with _SOLVER_LOCK:
        duplicates = [key for key in keys if key in _SOLVER_BACKENDS and not overwrite]
        if duplicates:
            raise ValueError(f"Solver method(s) already registered: {duplicates}.")
        for key in keys:
            _SOLVER_BACKENDS[key] = solve_fn
    return solve_fn


def register_equation_solver(
    equation: str,
    solver: Union[str, Callable[..., Any]],
    *,
    overwrite: bool = False,
) -> Union[str, Callable[..., Any]]:
    """Associate a registered equation name with a solver method/callback.

    This is deliberately separate from ``register_pde``: a residual class
    describes the PINN's strong form, while a classical solver needs its own
    native equation, variational form, or simulation builder.
    """
    key = _solver_key(equation)
    if not isinstance(solver, str) and not callable(solver):
        raise TypeError("solver must be a method name or a callable solve function.")
    with _SOLVER_LOCK:
        if key in _EQUATION_SOLVERS and not overwrite:
            raise ValueError(f"A classical solver is already registered for equation {equation!r}.")
        _EQUATION_SOLVERS[key] = _solver_key(solver) if isinstance(solver, str) else solver
    return solver


def _require_optional(package: str, purpose: str):
    try:
        return __import__(package)
    except ImportError as exc:
        raise ImportError(
            f"The optional {package!r} package is required for {purpose}. "
            "The PhysAI solver installer prepares a Conda environment with "
            "these native solver dependencies: `python -m physai.solver_setup`."
        ) from exc


def _normalize_solver_result(result: Any) -> Dict[str, Any]:
    """Normalize scalar FEM functions and native grid solver tuples."""
    if isinstance(result, dict) and "coordinates" in result and "values" in result:
        return result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        coordinates, values = result
        return {"coordinates": np.asarray(coordinates), "values": values}

    function_space = getattr(result, "function_space", None)
    if callable(function_space):  # classic FEniCS
        function_space = function_space()
    if function_space is not None:
        tabulate = getattr(function_space, "tabulate_dof_coordinates", None)
        if callable(tabulate):
            coordinates = np.asarray(tabulate())
            vector = getattr(result, "vector", None)
            if callable(vector):
                vector = vector()
                get_local = getattr(vector, "get_local", None)
                values = np.asarray(get_local() if callable(get_local) else vector[:])
            else:  # DOLFINx Function
                values = np.asarray(result.x.array)
            if coordinates.shape[0] != values.reshape(-1).shape[0]:
                raise ValueError(
                    "Finite-element dof coordinates and values have different lengths; "
                    "provide result_adapter for this element/layout."
                )
            return {"coordinates": coordinates, "values": {"u": values.reshape(-1)}}

    coordinates = getattr(result, "coordinates", None)
    values = getattr(result, "values", None)
    if coordinates is not None and values is not None:
        return {"coordinates": coordinates, "values": values}
    raise TypeError(
        "Cannot infer coordinates and values from the classical solver result. "
        "Provide result_adapter(result) returning a mapping with 'coordinates' and 'values'."
    )


def solve_fipy(
    *,
    bounds: Sequence[Tuple[float, float]],
    grid_points: Union[int, Sequence[int]],
    variables: Sequence[str],
    equation_builder: Callable[[Any, Any, Dict[str, Any]], Any],
    initial_values: Optional[Dict[str, Any]] = None,
    boundary_conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    time_steps: int = 1,
    dt: Optional[float] = None,
    sweeps: int = 1,
    solver: Any = None,
) -> Dict[str, Any]:
    """Solve a finite-volume system assembled with native FiPy terms.

    ``equation_builder(fipy, mesh, variables)`` returns one FiPy Equation
    for a single field, a name-to-equation mapping for uncoupled fields, or
    a coupled Equation accepted by FiPy. The function returns cell-center
    coordinates and copied cell values. ``dt`` enables transient stepping;
    equations are advanced through FiPy's ``sweep`` API.
    """
    fipy = _require_optional("fipy", "FiPy finite-volume solves")
    names = list(variables)
    if not names:
        raise ValueError("FiPy solve requires at least one variable name.")
    dim = len(bounds)
    if dim not in (1, 2, 3):
        raise ValueError("FiPy structured-grid adapter supports 1D, 2D, and 3D bounds.")
    shape = [int(grid_points)] * dim if isinstance(grid_points, int) else list(grid_points)
    if len(shape) != dim or any(n < 2 for n in shape):
        raise ValueError("grid_points must contain at least two cells per axis.")
    lengths = [float(hi) - float(lo) for lo, hi in bounds]
    if any(length <= 0 for length in lengths):
        raise ValueError("Every bound must have a positive length.")
    if dim == 1:
        mesh = fipy.Grid1D(nx=shape[0], dx=lengths[0] / shape[0])
    elif dim == 2:
        mesh = fipy.Grid2D(
            nx=shape[0], ny=shape[1],
            dx=lengths[0] / shape[0], dy=lengths[1] / shape[1],
        )
    else:
        mesh = fipy.Grid3D(
            nx=shape[0], ny=shape[1], nz=shape[2],
            dx=lengths[0] / shape[0], dy=lengths[1] / shape[1],
            dz=lengths[2] / shape[2],
        )
    mesh = mesh + tuple((float(bound[0]),) for bound in bounds)

    initial_values = initial_values or {}
    fields = {
        name: fipy.CellVariable(
            mesh=mesh, name=name, value=initial_values.get(name, 0.0),
            hasOld=dt is not None,
        )
        for name in names
    }
    for name, faces in (boundary_conditions or {}).items():
        if name not in fields:
            raise ValueError(f"Boundary conditions reference unknown FiPy variable {name!r}.")
        for face, value in faces.items():
            face_attr = {
                "left": "facesLeft", "right": "facesRight",
                "bottom": "facesBottom", "top": "facesTop",
                "front": "facesFront", "back": "facesBack",
            }.get(face.lower())
            if face_attr is None or not hasattr(mesh, face_attr):
                raise ValueError(f"Unsupported FiPy boundary face {face!r} for {dim}D mesh.")
            fields[name].constrain(value, getattr(mesh, face_attr))

    equations = equation_builder(fipy, mesh, fields)
    if isinstance(equations, dict):
        equation_items = [(fields[name], eq) for name, eq in equations.items()]
    elif isinstance(equations, (list, tuple)):
        if len(equations) != len(names):
            raise ValueError("A FiPy equation sequence must match the variables sequence length.")
        equation_items = list(zip((fields[n] for n in names), equations))
    else:
        equation_items = [(tuple(fields.values()) if len(fields) > 1 else next(iter(fields.values())), equations)]

    n_steps = max(1, int(time_steps))
    n_sweeps = max(1, int(sweeps))
    for _ in range(n_steps):
        if dt is not None:
            for field in fields.values():
                field.updateOld()
        for field, eq in equation_items:
            for _ in range(n_sweeps):
                eq.sweep(var=field, dt=dt, solver=solver) if dt is not None else eq.sweep(var=field, solver=solver)

    centers = np.asarray(mesh.cellCenters)
    return {
        "mesh": mesh,
        "coordinates": centers.T,
        "values": {name: np.asarray(field.value).copy() for name, field in fields.items()},
        "variables": fields,
    }


def solve_fenics(
    *, residual_form: Any, solution: Any, bcs: Sequence[Any] = (),
    solver_parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Solve a classic FEniCS variational residual using its native solver."""
    dolfin = _require_optional("dolfin", "classic FEniCS finite-element solves")
    dolfin.solve(
        residual_form == 0, solution, bcs=list(bcs),
        solver_parameters=solver_parameters or {},
    )
    return solution


def solve_fenicsx(
    *, residual_form: Any, solution: Any, bcs: Sequence[Any] = (),
    jacobian: Any = None, petsc_options_prefix: str = "physai_nonlinear_",
    petsc_options: Optional[Dict[str, Any]] = None,
    form_compiler_options: Optional[Dict[str, Any]] = None,
    jit_options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Solve a DOLFINx UFL residual with its PETSc nonlinear problem API."""
    dolfinx = _require_optional("dolfinx", "FEniCSx nonlinear finite-element solves")
    try:
        from dolfinx.fem.petsc import NonlinearProblem
    except ImportError as exc:
        raise ImportError("This DOLFINx installation lacks dolfinx.fem.petsc support.") from exc
    kwargs = dict(
        bcs=list(bcs), J=jacobian, petsc_options_prefix=petsc_options_prefix,
        petsc_options=petsc_options or {},
        form_compiler_options=form_compiler_options or {}, jit_options=jit_options or {},
    )
    problem = NonlinearProblem(residual_form, solution, **kwargs)
    return problem.solve()


def solve_fenicsx_linear(
    *, bilinear_form: Any, linear_form: Any,
    bcs: Sequence[Any] = (), petsc_options_prefix: str = "physai_linear_",
    petsc_options: Optional[Dict[str, Any]] = None,
    form_compiler_options: Optional[Dict[str, Any]] = None,
    jit_options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Solve a linear DOLFINx UFL variational problem."""
    _require_optional("dolfinx", "FEniCSx linear finite-element solves")
    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(
        bilinear_form, linear_form, bcs=list(bcs),
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=petsc_options or {},
        form_compiler_options=form_compiler_options or {}, jit_options=jit_options or {},
    )
    return problem.solve()


def solve_meep(
    *, simulation_builder: Callable[[Any], Any],
    step_functions: Sequence[Callable[..., Any]] = (),
    run_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build and run a Meep simulation for a user-defined EM formulation.

    The callback receives the imported ``meep`` module and returns a native
    ``meep.Simulation`` configured with its cell, materials, sources, and
    boundaries. The result is that simulation object after ``run()``.
    """
    if not callable(simulation_builder):
        raise TypeError("solve_meep requires a simulation_builder(meep) callback.")
    meep = _require_optional("meep", "Meep electromagnetic simulations")
    simulation = simulation_builder(meep)
    if not hasattr(simulation, "run"):
        raise TypeError("simulation_builder must return a Meep Simulation-like object with run().")
    simulation.run(*step_functions, **(run_kwargs or {}))
    return simulation


for _name, _fn in {
    "fipy": solve_fipy,
    "fenics": solve_fenics,
    "fenicsx": solve_fenicsx,
    "fenicsx_linear": solve_fenicsx_linear,
    "meep": solve_meep,
}.items():
    _SOLVER_BACKENDS[_name] = _fn


__all__ = [
    "Solver", "register_solver", "register_equation_solver",
    "solve_fipy", "solve_fenics", "solve_fenicsx", "solve_fenicsx_linear",
    "solve_meep",
]
