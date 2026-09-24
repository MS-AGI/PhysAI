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

  Built-in finite-difference equations share a masked-grid core. Other
  registered or caller-supplied residuals route through ``autosolve`` on
  the same arbitrary SDF geometry. ``solve_discrete_geometry`` remains
  available when callers want to supply a sparse, equation-specific
  discretization instead of meshless residual collocation.

Optional native adapters are imported only when called. FiPy accepts an
equation-builder callback and returns cell-center samples. FEniCS and
FEniCSx accept native variational forms; Meep accepts a simulation-builder
callback. Results outside the standard ``{"coordinates": ..., "values": ...}``
shape can be normalized with ``Solver.cross_validate(result_adapter=...)``.
"""
from __future__ import annotations

import logging
import re
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


def _cut_boundary_links(geometry: Geometry, points: np.ndarray, mask: np.ndarray,
                        grid_shape: Sequence[int], spacings: Sequence[float]):
    """Find SDF-projected boundary crossings for every masked-grid cut edge.

    Each record is ``(flat_node, axis, sign, neighbor_flat, fraction,
    boundary_point)``. ``fraction`` is the distance from the interior grid
    node to the boundary divided by the full grid spacing along that axis.
    The crossing is located by bisection on the actual SDF, so smooth and
    curved boundaries are represented on their zero level set rather than
    substituted at the first exterior grid node.
    """
    dims = len(grid_shape)
    n_total = len(points)
    flat_indices = np.arange(n_total, dtype=np.int64)
    multi = np.stack(np.unravel_index(flat_indices, grid_shape), axis=1)
    strides = np.asarray([int(np.prod(grid_shape[d + 1:])) for d in range(dims)], dtype=np.int64)
    records = []
    for axis in range(dims):
        for sign in (-1, 1):
            neighbor_coord = multi[:, axis] + sign
            valid = (neighbor_coord >= 0) & (neighbor_coord < grid_shape[axis])
            neighbor_flat = flat_indices + sign * strides[axis]
            safe_neighbor = np.clip(neighbor_flat, 0, n_total - 1)
            cut = mask & (~valid | ~mask[safe_neighbor])
            nodes = flat_indices[cut]
            if not len(nodes):
                continue
            neighbors = np.where(valid[cut], neighbor_flat[cut], -1).astype(np.int64)
            inside = points[nodes]
            outside = points[np.maximum(neighbors, 0)].copy()
            outside[neighbors < 0] = inside[neighbors < 0]
            outside[neighbors < 0, axis] += sign * spacings[axis]
            d_inside = geometry.distance(inside)
            d_outside = geometry.distance(outside)
            bracketed = (d_inside < 0.0) & (d_outside >= 0.0)
            lo = np.zeros(len(nodes), dtype=np.float64)
            hi = np.ones(len(nodes), dtype=np.float64)
            for _ in range(28):
                active = bracketed
                if not active.any():
                    break
                mid = 0.5 * (lo + hi)
                probe = inside + mid[:, None] * (outside - inside)
                d_mid = geometry.distance(probe)
                inside_mid = active & (d_mid <= 0.0)
                outside_mid = active & ~inside_mid
                lo[inside_mid] = mid[inside_mid]
                hi[outside_mid] = mid[outside_mid]
            fraction = np.where(bracketed, 0.5 * (lo + hi), 1.0)
            boundary = inside + fraction[:, None] * (outside - inside)
            records.extend(
                (int(node), axis, sign, int(neighbor), float(frac), point)
                for node, neighbor, frac, point in zip(nodes, neighbors, fraction, boundary)
            )
    return records


# ---------------------------------------------------------------------------
# Tau bookkeeping for Chebyshev (non-periodic) axes in Dedalus box solves
# ---------------------------------------------------------------------------
#
# A tau method removes as many degrees of freedom from the differential
# equations as there are boundary conditions, and hands them back as scalar
# (or boundary-basis) "tau" unknowns. Dedalus refuses a system that is not
# square, so the number of taus has to equal the number of boundary
# equations exactly. That number follows from the *spatial derivative order
# of each equation along the Chebyshev axis*, not from how many times a
# particular spelling appears in the equation text, and periodic (Fourier)
# axes need no taus at all.

_DERIVATIVE_TOKEN = re.compile(
    r"(?<![\w.])(?:(dx)(?:_x(\d+))?|d3\.(Differentiate|diff|grad|div|curl|lap|Laplacian))\s*\("
)


def _spatial_derivative_orders(expr: str, dims: int) -> List[int]:
    """Highest total spatial-derivative order of ``expr`` along each axis.

    Nesting adds orders (``dx_x0(dx_x0(u))`` is second order in ``x0``);
    separate terms do not (``dx(u) + dx(v)`` is first order). ``dt`` is not a
    spatial derivative. The generic ``d3.grad/div/curl/Differentiate`` calls
    count as first order and ``d3.lap/Laplacian`` as second order on every
    axis, since they do not name one.
    """
    opens = {}
    for m in _DERIVATIVE_TOKEN.finditer(expr):
        if m.group(1):
            axis = int(m.group(2)) if m.group(2) is not None else 0
            weight = 1
        else:
            axis = None
            weight = 2 if m.group(3) in ("lap", "Laplacian") else 1
        opens[m.end() - 1] = (axis, weight)

    current = [0] * dims
    best = [0] * dims
    stack: List[Any] = []

    def _shift(info, sign):
        axis, weight = info
        for a in (range(dims) if axis is None else ([axis] if axis < dims else [])):
            current[a] += sign * weight

    for pos, ch in enumerate(expr):
        if ch == "(":
            info = opens.get(pos)
            stack.append(info)
            if info:
                _shift(info, +1)
                for a in range(dims):
                    best[a] = max(best[a], current[a])
        elif ch == ")":
            info = stack.pop() if stack else None
            if info:
                _shift(info, -1)
    return best


def _split_equation(eq: str) -> Tuple[str, str]:
    """Split ``"LHS = RHS"`` at its top-level ``=`` (ignoring ``x0='left'``)."""
    depth = 0
    for i, ch in enumerate(eq):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = eq[i - 1] if i > 0 else ""
            nxt = eq[i + 1] if i + 1 < len(eq) else ""
            if prev in "=<>!" or nxt == "=":
                continue
            return eq[:i].rstrip(), eq[i + 1:].lstrip()
    raise ValueError(f"Equation {eq!r} must have the form 'LHS = RHS'.")


def _plan_taus(equations: Sequence[str], n_bcs: int, domain_types: Sequence[str]):
    """Decide where taus go: returns ``(lift_axis, tau_counts_per_equation)``.

    ``lift_axis`` is the index of the Chebyshev axis the boundary conditions
    act on (or None when there is nothing to lift). Each equation receives as
    many taus as its derivative order along that axis, and the total must
    equal ``n_bcs`` -- otherwise the system cannot be square and a clear
    error is raised here instead of an opaque one from Dedalus.
    """
    dims = len(domain_types)
    orders = [_spatial_derivative_orders(eq, dims) for eq in equations]
    cheb_axes = [i for i, t in enumerate(domain_types) if t == "chebyshev"]
    active = [a for a in cheb_axes if any(o[a] > 0 for o in orders)]
    if len(active) > 1:
        raise NotImplementedError(
            f"solve_box supports boundary conditions along one Chebyshev axis "
            f"(with any number of periodic Fourier axes); the equations "
            f"differentiate along {len(active)} Chebyshev axes "
            f"({['x%d' % a for a in active]}). Use Fourier bases for the other "
            "axes, or solve this problem with Dedalus directly."
        )
    if active:
        lift_axis: Optional[int] = active[0]
        counts = [o[lift_axis] for o in orders]
    else:
        lift_axis, counts = None, [0] * len(equations)
    total = sum(counts)
    if total != n_bcs:
        where = f"along Chebyshev axis x{lift_axis}" if lift_axis is not None else "(no Chebyshev axis is differentiated)"
        raise ValueError(
            f"Boundary conditions do not match the equations: their spatial "
            f"derivative orders {counts} call for {total} boundary condition(s) "
            f"{where}, but {n_bcs} were given. Periodic (Fourier) axes take no "
            "boundary conditions; each Chebyshev derivative order needs exactly one."
        )
    return lift_axis, counts


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
        # Same bare `dx` alias that solve_box provides for 1-D problems.
        if dims == 1:
            self.local_env['dx'] = self.local_env[f"dx_{axis_names[0]}"]

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
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if not np.isfinite(stop_time) or stop_time < 0:
            raise ValueError("stop_time must be finite and non-negative.")
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("variables must contain at least one unique field name.")
        if not equations:
            raise ValueError("equations must contain at least one equation.")
        variables = [variables] if isinstance(variables, str) else list(variables)
        equations = [equations] if isinstance(equations, str) else list(equations)
        bcs = [bcs] if isinstance(bcs, str) else list(bcs)
        if not isinstance(ics, dict):
            raise TypeError("ics must map variable names to initial-condition callables.")
        if any(not callable(fn) for fn in ics.values()):
            raise TypeError("Every initial-condition value in ics must be callable.")
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
        if any(int(n) != n or int(n) < 2 for n in grid_points_list):
            raise ValueError("grid_points must be an integer of at least 2 on every axis.")
        for axis_bounds in bounds_list:
            if (len(axis_bounds) != 2 or not np.all(np.isfinite(axis_bounds))
                    or axis_bounds[1] <= axis_bounds[0]):
                raise ValueError("Each bounds entry must be a finite increasing (lower, upper) pair.")

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

        # Tau unknowns: one per boundary condition, lifted into the equations
        # that carry the highest derivative along the (single) Chebyshev
        # boundary axis; none for periodic axes. See ``_plan_taus``.
        lift_axis, tau_counts = _plan_taus(equations, len(bcs), domain_types)
        tau_bases = tuple(
            b for i, b in enumerate(bases) if i != lift_axis
        ) if lift_axis is not None else ()
        tau_fields = []
        modified_equations = []
        for idx, (eq_str, n_taus) in enumerate(zip(equations, tau_counts)):
            if n_taus == 0:
                modified_equations.append(eq_str)
                continue
            lifts = []
            for t_idx in range(n_taus):
                tau_name = f"tau_{idx}_{t_idx}"
                tau_field = dist.Field(name=tau_name, bases=tau_bases)
                local_env[tau_name] = tau_field
                tau_fields.append(tau_field)
                lifts.append(f"d3.lift({tau_name}, lift_basis, {-t_idx - 1})")
            lhs, rhs = _split_equation(eq_str)
            modified_equations.append(f"{lhs} + {' + '.join(lifts)} = {rhs}")

        eval_env = dict(local_env)
        eval_env["bases"] = bases
        eval_env["chebyshev_bases"] = [b for b, t in zip(bases, domain_types) if t == "chebyshev"]
        if lift_axis is not None:
            eval_env["lift_basis"] = bases[lift_axis]
        # `left(f)` / `right(f)` shorthand for f(x<k>='left'/'right') on the
        # boundary axis (the documented `bcs=["left(u) = 0", ...]` spelling).
        cheb_axes = [i for i, t in enumerate(domain_types) if t == "chebyshev"]
        bc_axis = lift_axis if lift_axis is not None else (cheb_axes[0] if cheb_axes else None)
        if bc_axis is not None:
            eval_env.setdefault("left", lambda f, _a=axis_names[bc_axis]: f(**{_a: "left"}))
            eval_env.setdefault("right", lambda f, _a=axis_names[bc_axis]: f(**{_a: "right"}))

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
        pde: Optional[str] = None,
        bc_value: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        grid_points: Union[int, Sequence[int]] = 64,
        pde_params: Optional[Dict[str, float]] = None,
        source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ic: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dt: float = 1e-3,
        n_steps: int = 1000,
        array_module: str = "numpy",
        dtype=np.float64,
        residual: Any = None,
        backend: Any = "torch",
        residual_kwargs: Optional[Dict[str, Any]] = None,
        boundary_conditions: Any = None,
        time_domain: Optional[Tuple[float, float]] = None,
        ic_points: Any = None,
        ic_values: Any = None,
        data_points: Any = None,
        data_values: Any = None,
        n_output: Optional[int] = None,
        n_collocation: Optional[int] = None,
        n_boundary: Optional[int] = None,
        max_nfev: int = 200,
        tolerance: float = 1e-7,
        max_unknowns: int = 192,
        rbf_epsilon: Optional[float] = None,
        rbf_ridge: Optional[float] = None,
        residual_balance: bool = True,
        diff_mode: str = "reverse",
        seed: Optional[int] = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve a PDE on an arbitrary ``physai.geometry.Geometry``. The built-in
        scalar linear equations use sparse embedded-boundary finite
        differences; registered PDE names and caller-supplied residuals use
        the general AutoSolve collocation path.

        Parameters
        ----------
        geometry     : any physai.geometry.Geometry (box, ball, CSG
                       combination, custom SDF, or a mesh loaded via
                       geometry_from_file).
        pde          : one of the built-in equations, or any registered PDE
                       name when using AutoSolve. For an unregistered PDE,
                       pass its residual callable through ``residual``.
        bc_value     : callable mapping an (N, dim) array of physical
                       coordinates to prescribed Dirichlet values. On
                       curved boundaries it is evaluated at each SDF-zero
                       crossing along a cut grid edge.
        grid_points  : grid resolution, either one int (applied to every
                       axis) or a per-axis sequence.
        source       : callable (N, dim) -> (N,) RHS forcing; zero if None.
        ic           : required for pde="heat"; callable (N, dim) -> (N,)
                       initial condition on interior nodes.
        array_module : "numpy" (default) or "cupy" (GPU; requires cupy).
        residual     : optional ``residual(model_fn, points)`` callable or
                       PDEResidual instance; selects the general AutoSolve
                       path for arbitrary PDEs and geometries.
        boundary_conditions : arbitrary-geometry boundary conditions for
                       AutoSolve. ``bc_value`` is converted to a Dirichlet
                       condition when this is omitted.

        Returns
        -------
        (points, mask, values) where ``points`` is the (Ntotal, dim) full
        regular-grid coordinate array, ``mask`` is a boolean (Ntotal,)
        array that is True where the node is inside ``geometry`` (i.e.
        where ``values`` is meaningful — exterior nodes are filled with
        NaN), and ``values`` is the solved field, same shape as ``mask``.
        """
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a physai.geometry.Geometry.")
        if pde is not None and not isinstance(pde, str):
            raise TypeError("pde must be a registered PDE name or None.")
        pde = pde.strip().lower() if pde else None
        if pde_params is None:
            pde_params = {}
        elif not isinstance(pde_params, dict):
            raise TypeError("pde_params must be a mapping.")

        # The sparse finite-difference kernels below are specialized fast
        # paths. Every other residual goes through the registry-independent
        # meshless solver, which accepts arbitrary Geometry SDFs and any
        # backend-differentiable residual callable.
        if residual is not None or pde not in Solver._SUPPORTED_PDES:
            if source is not None:
                raise ValueError(
                    "For AutoSolve, put source terms into the residual callable; "
                    "the legacy source callback is only used by built-in finite differences."
                )
            from physai.solvers.auto_solver import autosolve
            from physai.backends import get_backend
            from physai.geometry import BoundaryConditionSet, everywhere

            selected_backend = get_backend(backend) if isinstance(backend, str) else backend
            if residual is None and pde is None:
                raise ValueError("Supply either a registered pde name or a residual callable.")
            if boundary_conditions is not None and bc_value is not None:
                raise ValueError("Pass boundary_conditions or bc_value, not both.")
            if boundary_conditions is None and bc_value is not None:
                if not callable(bc_value):
                    raise TypeError("bc_value must be callable.")
                boundary_conditions = BoundaryConditionSet(geometry)
                boundary_conditions.add(
                    "dirichlet", value=bc_value, region=everywhere, name="dirichlet_all"
                )
            if ic is not None:
                if not callable(ic):
                    raise TypeError("ic must be callable.")
                if time_domain is None and pde == "heat":
                    time_domain = (0.0, float(dt) * int(n_steps))
                if time_domain is None:
                    raise ValueError("An initial-condition callable requires time_domain for AutoSolve.")
                if ic_points is None:
                    rng = np.random.default_rng(seed)
                    ic_points = geometry.sample_interior(n_collocation or 32, rng=rng)
                if ic_values is None:
                    ic_array = np.asarray(ic_points)
                    ic_values = ic(ic_array[:, :geometry.dim] if ic_array.ndim == 2 else ic_array)
            elif (ic_points is None) != (ic_values is None):
                raise ValueError("ic_points and ic_values must be supplied together.")
            auto_result = autosolve(
                residual,
                geometry,
                backend=selected_backend,
                pde=pde,
                residual_kwargs=residual_kwargs or pde_params,
                n_output=n_output,
                time_domain=time_domain,
                boundary_conditions=boundary_conditions,
                ic_points=ic_points,
                ic_values=ic_values,
                data_points=data_points,
                data_values=data_values,
                n_collocation=n_collocation,
                n_boundary=n_boundary,
                rbf_epsilon=rbf_epsilon,
                rbf_ridge=rbf_ridge,
                max_nfev=max_nfev,
                tolerance=tolerance,
                max_unknowns=max_unknowns,
                residual_balance=residual_balance,
                diff_mode=diff_mode,
                seed=seed,
            )
            auto_coordinates = np.asarray(auto_result["coordinates"])
            auto_fields = auto_result["values"]
            auto_values = np.column_stack(list(auto_fields.values()))
            if auto_values.shape[1] == 1:
                auto_values = auto_values[:, 0]
            return auto_coordinates, np.ones(len(auto_coordinates), dtype=bool), auto_values

        if bc_value is None or not callable(bc_value):
            raise TypeError("Built-in geometry solves require a callable bc_value.")
        dims = geometry.dim
        n_per_axis = _as_axis_list(grid_points, dims, "grid_points")
        if any(int(n) != n or int(n) < 2 for n in n_per_axis):
            raise ValueError("grid_points must be an integer of at least 2 on every axis.")
        if not callable(bc_value):
            raise TypeError("bc_value must be callable.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")

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

        # 2. Exclude boundary nodes from the unknowns. Cut edges are then
        # located on the SDF zero set, so curved boundaries enter the
        # finite-difference stencil at their actual position rather than at
        # the first exterior grid node.
        mask = np.asarray(geometry.contains(points, tol=-1e-9)).reshape(-1).astype(bool)
        n_total = points.shape[0]
        if not np.any(mask):
            raise ValueError("solve_geometry: no grid nodes fall inside `geometry` — increase grid_points.")

        # Map full flat index -> unknown index (only for interior/domain nodes).
        unknown_index = -np.ones(n_total, dtype=np.int64)
        unknown_index[mask] = np.arange(int(mask.sum()))
        n_unknown = int(mask.sum())

        strides = np.array([int(np.prod(grid_shape[d + 1:])) for d in range(dims)], dtype=np.int64)
        cut_links = {
            (node, axis, sign): (neighbor, fraction, boundary)
            for node, axis, sign, neighbor, fraction, boundary
            in _cut_boundary_links(geometry, points, mask, grid_shape, spacings)
        }

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        rhs = np.zeros(n_unknown, dtype=dtype)

        flat_idx_domain = np.nonzero(mask)[0]
        for flat_idx in flat_idx_domain:
            row = int(unknown_index[flat_idx])
            multi_idx = np.unravel_index(flat_idx, grid_shape)
            diag = 0.0
            for axis in range(dims):
                side_data = {}
                for sign in (-1, 1):
                    neighbor_multi = list(multi_idx)
                    neighbor_multi[axis] += sign
                    if 0 <= neighbor_multi[axis] < grid_shape[axis]:
                        neighbor_flat = int(flat_idx + sign * strides[axis])
                    else:
                        neighbor_flat = -1
                    if neighbor_flat >= 0 and mask[neighbor_flat]:
                        side_data[sign] = (float(spacings[axis]), neighbor_flat, None)
                    else:
                        neighbor_flat, fraction, boundary = cut_links[(int(flat_idx), axis, sign)]
                        side_data[sign] = (float(spacings[axis] * fraction), neighbor_flat, boundary)

                h_minus = side_data[-1][0]
                h_plus = side_data[1][0]
                h_sum = h_minus + h_plus
                diag += 2.0 * (1.0 / h_minus + 1.0 / h_plus) / h_sum
                for sign, h_side in ((-1, h_minus), (1, h_plus)):
                    neighbor_flat, _fraction, boundary = side_data[sign]
                    coefficient = -2.0 / (h_sum * h_side)
                    if boundary is None:
                        rows.append(row)
                        cols.append(int(unknown_index[neighbor_flat]))
                        data.append(coefficient)
                    else:
                        boundary_value = np.asarray(bc_value(boundary.reshape(1, -1))).reshape(-1)
                        if boundary_value.size != 1 or not np.isfinite(boundary_value[0]):
                            raise ValueError("bc_value must return one finite scalar per boundary point.")
                        g = float(boundary_value[0])
                        rhs[row] -= coefficient * g
            rows.append(row)
            cols.append(row)
            data.append(diag)

        L = sp.coo_matrix((data, (rows, cols)), shape=(n_unknown, n_unknown)).tocsr()

        src_vals = np.zeros(n_unknown, dtype=dtype)
        if source is not None:
            src_vals = np.asarray(source(points[mask])).reshape(-1).astype(dtype)
            if src_vals.shape != (n_unknown,) or not np.all(np.isfinite(src_vals)):
                raise ValueError(f"source must return {n_unknown} finite scalar values on the interior grid.")

        values_full = np.full(n_total, np.nan, dtype=dtype)

        if pde == "poisson":
            b = rhs + src_vals
            A = L if array_module == "numpy" else xsp.csr_matrix(L)
            u = xspla.spsolve(A, xp.asarray(b) if array_module == "cupy" else b)
            u = np.asarray(u.get()) if array_module == "cupy" else np.asarray(u)
            values_full[mask] = u

        elif pde == "helmholtz":
            k = pde_params.get("k")
            if k is None or not np.isfinite(k):
                raise ValueError("pde='helmholtz' requires a finite pde_params={'k': ...}")
            A_np = (L - (k ** 2) * sp.identity(n_unknown, dtype=dtype, format="csr"))
            b = rhs + src_vals
            A = A_np if array_module == "numpy" else xsp.csr_matrix(A_np)
            u = xspla.spsolve(A, xp.asarray(b) if array_module == "cupy" else b)
            u = np.asarray(u.get()) if array_module == "cupy" else np.asarray(u)
            values_full[mask] = u

        else:  # "heat"
            alpha = pde_params.get("alpha")
            if alpha is None or not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("pde='heat' requires a finite positive pde_params={'alpha': ...}")
            if ic is None or not callable(ic):
                raise ValueError("pde='heat' requires an initial condition `ic`")
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("pde='heat' requires a finite positive dt.")
            if int(n_steps) != n_steps or n_steps < 1:
                raise ValueError("pde='heat' requires a positive integer n_steps.")
            u = np.asarray(ic(points[mask])).reshape(-1).astype(dtype)
            if u.shape != (n_unknown,) or not np.all(np.isfinite(u)):
                raise ValueError(f"ic must return {n_unknown} finite scalar values on the interior grid.")
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
        custom_kwargs = dict(kwargs)
        if geometry is not None:
            custom_kwargs["geometry"] = geometry  # custom solvers see it too
        if callable(selected):
            return selected(**custom_kwargs)
        default_method = (
            "discrete_geometry"
            if geometry is not None and ("assembler" in kwargs or "residual_fn" in kwargs)
            else ("geometry" if geometry is not None else "dedalus")
        )
        selected = _solver_key(selected or default_method)

        if selected == "autosolve":
            from physai.solvers.auto_solver import autosolve
            if geometry is None:
                raise ValueError("method='autosolve' requires a Geometry instance.")
            return autosolve(geometry=geometry, **kwargs)

        if selected in {"dedalus", "geometry"}:
            if geometry is not None:
                if "assembler" in kwargs or "residual_fn" in kwargs:
                    return solve_discrete_geometry(**custom_kwargs)
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
        return solve_fn(**custom_kwargs)

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
        if (isinstance(spatial_dims, bool) or int(spatial_dims) != spatial_dims
                or spatial_dims < 1):
            raise ValueError("spatial_dims must be a positive integer.")
        if coordinates.shape[1] not in (spatial_dims, spatial_dims + 1):
            raise ValueError(
                f"Classical solver coordinates must have {spatial_dims} spatial columns "
                f"and at most one time column; got {coordinates.shape}."
            )
        if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
            raise ValueError("Classical solver coordinates must be non-empty and finite.")

        fields = normalized["values"]
        if not isinstance(fields, dict):
            fields = {"u": fields}
        names = ([variables] if isinstance(variables, str) else list(variables)) if variables is not None else list(fields)
        if not names:
            raise ValueError("No classical solution fields were returned for comparison.")

        if eval_time is None and solver_kwargs:
            # Evaluate the model at the time the classical solver actually
            # stopped at, not blindly at the end of the training window.
            if solver_kwargs.get("stop_time") is not None:
                eval_time = float(solver_kwargs["stop_time"])
            elif str(solver_kwargs.get("pde", "")).lower() == "heat":
                eval_time = float(solver_kwargs.get("n_steps", 1000)) * float(solver_kwargs.get("dt", 1e-3))
        comparison_time = eval_time
        if comparison_time is None and time_domain is not None:
            comparison_time = time_domain[1]
        if comparison_time is not None and not np.isfinite(comparison_time):
            raise ValueError("eval_time/comparison time must be finite.")
        query_points = coordinates
        # Keep per-point times from a transient result. Append a single
        # comparison time only for stationary spatial grids.
        if comparison_time is not None and coordinates.shape[1] == spatial_dims:
            query_points = np.concatenate(
                [coordinates, np.full((len(coordinates), 1), comparison_time)], axis=1
            )
        prediction = np.asarray(backend.to_numpy(model_fn(backend.tensor(query_points))))
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if prediction.ndim != 2 or prediction.shape[0] != len(coordinates):
            raise ValueError(
                "model_fn must return values with shape [number_of_coordinates, n_outputs]; "
                f"got {prediction.shape}."
            )

        errors: Dict[str, Union[int, float]] = {}
        for position, name in enumerate(names):
            if name not in fields and not (len(fields) == 1 and len(names) == 1):
                raise KeyError(f"Classical solver result has no field {name!r}.")
            col = output_index[name] if output_index is not None and name in output_index else position
            if col < 0 or col >= prediction.shape[1]:
                raise IndexError(
                    f"Model output column for {name!r} is {col}, but the model returned "
                    f"{prediction.shape[1]} output(s)."
                )
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


def solve_discrete_geometry(
    *,
    geometry: Geometry,
    grid_points: Union[int, Sequence[int]] = 64,
    time_domain: Optional[Tuple[float, float]] = None,
    time_points: int = 1,
    assembler: Optional[Callable[[Dict[str, Any]], Tuple[Any, Any]]] = None,
    residual_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    jacobian_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], Any]] = None,
    initial_guess: Any = 0.0,
    n_fields: int = 1,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a user-discretized classical PDE system on any ``Geometry``.

    This is the general classical extension point when no built-in
    discretization exists. It handles arbitrary signed-distance geometries,
    maps grid nodes to compact unknown indices, and solves either a linear
    sparse system or a nonlinear system with Newton iterations. The caller
    supplies the equation-specific discretization:

    * ``assembler(context) -> (A, b)`` represents ``A u = b``.
    * ``residual_fn(u, context) -> r`` and ``jacobian_fn(u, context) -> J``
      provide the residual and sparse Jacobian for Newton's method.

    ``context`` contains ``geometry``, full ``points`` and ``mask``,
    ``interior_points``, ``unknown_index`` (full-grid index to compact
    unknown index, -1 outside), ``grid_shape``, ``spacings``, and
    Optional ``time_domain``/``time_points`` append a final time coordinate
    to each point and expand the mask over a space-time grid. ``boundary_links``
    contains tuples of
    ``(node, axis, sign, neighbor, fraction, boundary_point)`` and the
    projected boundary points/normals are also available separately. The
    unknown vector is flattened in point-major order;
    the returned values have shape ``(Ngrid,)`` for one field or
    ``(Ngrid, n_fields)`` for a system. Boundary conditions and the PDE's
    discretization are the callback's responsibility. This keeps the core
    classical solver equation-agnostic without pretending it can infer a
    numerical method from a PDE name.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a physai.geometry.Geometry.")
    if (assembler is None) == (residual_fn is None):
        raise ValueError("Pass exactly one of `assembler` or `residual_fn`.")
    if residual_fn is not None and jacobian_fn is None:
        raise ValueError("Nonlinear solves require `jacobian_fn` for Newton iterations.")
    if assembler is not None and not callable(assembler):
        raise TypeError("assembler must be callable.")
    if residual_fn is not None and (not callable(residual_fn) or not callable(jacobian_fn)):
        raise TypeError("residual_fn and jacobian_fn must be callable.")
    if (isinstance(n_fields, bool) or int(n_fields) != n_fields or n_fields < 1
            or isinstance(max_iterations, bool) or int(max_iterations) != max_iterations
            or max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("n_fields/max_iterations must be positive and tolerance must be > 0.")

    dims = geometry.dim
    n_per_axis = _as_axis_list(grid_points, dims, "grid_points")
    if any(int(n) != n or int(n) < 2 for n in n_per_axis):
        raise ValueError("grid_points must be an integer of at least 2 on every axis.")
    axes = [np.linspace(lo, hi, int(n)) for (lo, hi), n in zip(geometry.bounds, n_per_axis)]
    spatial_spacings = np.asarray([axis[1] - axis[0] for axis in axes], dtype=np.float64)
    spatial_mesh = np.meshgrid(*axes, indexing="ij")
    spatial_shape = spatial_mesh[0].shape
    spatial_points = np.stack([axis_grid.reshape(-1) for axis_grid in spatial_mesh], axis=-1).astype(dtype)
    spatial_mask = np.asarray(geometry.contains(spatial_points, tol=-1e-9), dtype=bool).reshape(-1)
    if not spatial_mask.any():
        raise ValueError("solve_discrete_geometry: no grid nodes fall inside `geometry`; increase grid_points.")
    if time_domain is not None:
        if isinstance(time_points, bool) or int(time_points) != time_points or time_points < 2:
            raise ValueError("time_points must be an integer of at least 2 when time_domain is given.")
        try:
            interval = np.asarray(time_domain, dtype=np.float64)
        except (TypeError, ValueError):
            interval = np.asarray([], dtype=np.float64)
        if (interval.shape != (2,) or not np.all(np.isfinite(interval))
                or interval[1] <= interval[0]):
            raise ValueError("time_domain must be a finite increasing (start, stop) interval.")
        t0, t1 = map(float, interval)
        times = np.linspace(t0, t1, int(time_points), dtype=dtype)
        points = np.concatenate([
            np.repeat(spatial_points, len(times), axis=0),
            np.tile(times, len(spatial_points))[:, None],
        ], axis=1)
        mask = np.repeat(spatial_mask, len(times))
        grid_shape = spatial_shape + (len(times),)
        spacings = np.concatenate([spatial_spacings, [times[1] - times[0]]])
    else:
        if time_points != 1:
            raise ValueError("time_points > 1 requires `time_domain`.")
        times = None
        points = spatial_points
        mask = spatial_mask
        grid_shape = spatial_shape
        spacings = spatial_spacings
    unknown_index = -np.ones(len(points), dtype=np.int64)
    unknown_index[mask] = np.arange(int(mask.sum()))
    spatial_links = _cut_boundary_links(geometry, spatial_points, spatial_mask, spatial_shape, spatial_spacings)
    if times is None:
        boundary_links = spatial_links
    else:
        boundary_links = [
            (node * len(times) + time_idx, axis, sign,
             neighbor * len(times) + time_idx if neighbor >= 0 else -1,
             fraction, np.concatenate([boundary, [times[time_idx]]]))
            for node, axis, sign, neighbor, fraction, boundary in spatial_links
            for time_idx in range(len(times))
        ]
    boundary_points = (
        np.stack([entry[5] for entry in boundary_links]).astype(dtype)
        if boundary_links else np.empty((0, dims + (1 if times is not None else 0)), dtype=dtype)
    )
    boundary_normals = geometry.normal(boundary_points[:, :dims]) if len(boundary_points) else np.empty((0, dims), dtype=dtype)
    context = {
        "geometry": geometry,
        "points": points,
        "mask": mask,
        "interior_points": points[mask],
        "unknown_index": unknown_index,
        "grid_shape": grid_shape,
        "spacings": spacings,
        "boundary_links": boundary_links,
        "boundary_points": boundary_points,
        "boundary_normals": boundary_normals,
        "spatial_points": spatial_points,
        "spatial_mask": spatial_mask,
        "time_values": times,
        "time_domain": time_domain,
        "initial_point_indices": np.arange(len(spatial_points)) * (len(times) if times is not None else 1),
        "initial_unknown_indices": unknown_index[
            np.arange(len(spatial_points)) * (len(times) if times is not None else 1)
        ],
        "n_fields": int(n_fields),
    }
    n_unknown = int(mask.sum()) * int(n_fields)
    if assembler is not None:
        assembled = assembler(context)
        if not isinstance(assembled, (tuple, list)) or len(assembled) != 2:
            raise TypeError("assembler(context) must return (sparse_matrix, rhs).")
        matrix, rhs = assembled
        matrix = sp.csr_matrix(matrix, dtype=dtype)
        rhs = np.asarray(rhs, dtype=dtype).reshape(-1)
        if matrix.shape != (n_unknown, n_unknown) or rhs.shape != (n_unknown,):
            raise ValueError(
                f"assembler must return matrix ({n_unknown}, {n_unknown}) and "
                f"RHS ({n_unknown},); got {matrix.shape} and {rhs.shape}."
            )
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
            raise ValueError("assembler returned non-finite matrix or RHS values.")
        solution = np.asarray(spla.spsolve(matrix, rhs), dtype=dtype).reshape(-1)
    else:
        initial_arr = np.asarray(initial_guess, dtype=dtype)
        if initial_arr.ndim == 0:
            initial = np.full(n_unknown, initial_arr.item(), dtype=dtype)
        elif initial_arr.shape == (int(mask.sum()), n_fields) or initial_arr.size == n_unknown:
            initial = initial_arr.reshape(-1).copy()
        else:
            raise ValueError("initial_guess must be scalar or match the compact unknown shape.")

        solution = initial
        for _ in range(max_iterations):
            residual = np.asarray(residual_fn(solution, context), dtype=dtype).reshape(-1)
            if residual.shape != (n_unknown,) or not np.all(np.isfinite(residual)):
                raise ValueError("residual_fn must return a finite vector matching the unknown count.")
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if norm <= tolerance:
                break
            jacobian = sp.csr_matrix(jacobian_fn(solution, context), dtype=dtype)
            if jacobian.shape != (n_unknown, n_unknown) or not np.all(np.isfinite(jacobian.data)):
                raise ValueError("jacobian_fn must return a finite sparse Jacobian of the system size.")
            step = np.asarray(spla.spsolve(jacobian, -residual), dtype=dtype).reshape(-1)
            if not np.all(np.isfinite(step)):
                raise RuntimeError("Newton solve produced a non-finite update.")
            damping = 1.0
            accepted = False
            while damping >= 1.0 / 128.0:
                candidate = solution + damping * step
                candidate_residual = np.asarray(residual_fn(candidate, context), dtype=dtype).reshape(-1)
                if (candidate_residual.shape == (n_unknown,)
                        and np.all(np.isfinite(candidate_residual))
                        and np.linalg.norm(candidate_residual, ord=np.inf) < norm):
                    solution = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"Newton iteration stalled at residual norm {norm:.3e}; "
                    "check the discretization, Jacobian, or initial_guess."
                )
        else:
            final_residual = np.asarray(residual_fn(solution, context), dtype=dtype).reshape(-1)
            if final_residual.shape != (n_unknown,) or not np.all(np.isfinite(final_residual)):
                raise ValueError("residual_fn must return a finite vector matching the unknown count.")
            final_norm = float(np.linalg.norm(final_residual, ord=np.inf))
            if final_norm > tolerance:
                raise RuntimeError(
                    f"Newton iteration did not converge within {max_iterations} steps "
                    f"(residual norm {final_norm:.3e}, tolerance {tolerance:.3e})."
                )

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("The discrete system did not produce a finite solution.")
    values = np.full((len(points), int(n_fields)), np.nan, dtype=dtype)
    values[mask] = solution.reshape(int(mask.sum()), int(n_fields))
    return points, mask, values[:, 0] if n_fields == 1 else values


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
    if isinstance(result, tuple) and len(result) == 3:  # Solver.solve_geometry
        points, mask, values = result
        mask_arr = np.asarray(mask)
        if mask_arr.dtype == bool:
            selected_values = np.asarray(values)[mask_arr]
            if selected_values.ndim == 1:
                named_values = {"u": selected_values}
            else:
                named_values = {
                    f"u{i}": selected_values[:, i]
                    for i in range(selected_values.shape[1])
                }
            return {
                "coordinates": np.asarray(points)[mask_arr],
                "values": named_values,
            }
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        coordinates, values = result
        if isinstance(coordinates, (list, tuple)):  # N-D Dedalus: one grid per axis
            axes = [np.asarray(a).squeeze().reshape(-1) for a in coordinates]
            mesh = np.meshgrid(*axes, indexing="ij")
            coordinates = np.stack([m.reshape(-1) for m in mesh], axis=1)
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
    shape = [int(grid_points)] * dim if isinstance(grid_points, (int, np.integer)) else list(grid_points)
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




def autosolve(residual: Any = None, geometry: Optional[Geometry] = None, **kwargs):
    """Solve a residual with the classical meshless RBF collocation pipeline."""
    from physai.solvers.auto_solver import autosolve as _autosolve
    return _autosolve(residual, geometry, **kwargs)


for _name, _fn in {
    "autosolve": autosolve,
    "discrete_geometry": solve_discrete_geometry,
    "sparse_geometry": solve_discrete_geometry,
    "fipy": solve_fipy,
    "fenics": solve_fenics,
    "fenicsx": solve_fenicsx,
    "fenicsx_linear": solve_fenicsx_linear,
    "meep": solve_meep,
}.items():
    if _fn is not None:
        _SOLVER_BACKENDS[_name] = _fn


__all__ = [
    "Solver", "register_solver", "register_equation_solver",
    "solve_discrete_geometry",
    "autosolve",
    "solve_fipy", "solve_fenics", "solve_fenicsx", "solve_fenicsx_linear",
    "solve_meep",
]
