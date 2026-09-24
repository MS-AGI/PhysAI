"""Registry-independent classical meshless PDE collocation.

``autosolve`` consumes the same ``residual(model_fn, points)`` contract as
``PDEResidual`` but uses a radial-basis collocation field and a direct
nonlinear least-squares solve. It does not build or train a neural network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.linalg import solve as dense_solve

from physai.backends import get_backend
from physai.core.auto_optimizer import get_pde_meta
from physai.geometry import BoundaryConditionSet, Geometry


@dataclass(frozen=True)
class AutoSolverConfig:
    """Inspectable numerical settings selected for one residual/geometry pair."""

    n_collocation: int
    n_boundary: int
    rbf_epsilon: float
    rbf_ridge: float
    max_nfev: int
    tolerance: float
    max_unknowns: int
    coordinate_scales: Tuple[float, ...]
    residual_balance: bool
    adaptive_ridge: bool
    selection_reason: str


class AutoSolverOptimizer:
    """Select stable, budgeted RBF settings from problem and optional metadata.

    The optimizer is deliberately registry-optional. When metadata exists it
    uses PDE order, stiffness, nonlinearity, and spectral bias; otherwise it
    falls back to geometry dimension and the requested output count. The
    point budget is kept below the dense nonlinear solve's practical limit.
    """

    def analyse(
        self,
        geometry: Geometry,
        *,
        n_output: int,
        pde: Optional[str] = None,
        residual: Any = None,
        time_domain: Optional[Tuple[float, float]] = None,
        has_boundary_conditions: bool = False,
        n_collocation: Optional[int] = None,
        n_boundary: Optional[int] = None,
        rbf_epsilon: Optional[float] = None,
        rbf_ridge: Optional[float] = None,
        max_nfev: int = 200,
        tolerance: float = 1e-7,
        max_unknowns: int = 192,
        residual_balance: bool = True,
    ) -> AutoSolverConfig:
        if isinstance(n_output, (bool, np.bool_)) or int(n_output) != n_output or n_output < 1:
            raise ValueError("n_output must be at least 1.")
        if (isinstance(max_nfev, (bool, np.bool_)) or int(max_nfev) != max_nfev or max_nfev < 1
                or not np.isfinite(tolerance) or tolerance <= 0
                or (rbf_ridge is not None and (not np.isfinite(rbf_ridge) or rbf_ridge < 0))):
            raise ValueError("max_nfev/tolerance must be positive and rbf_ridge non-negative.")
        if int(max_unknowns) != max_unknowns or max_unknowns < 2:
            raise ValueError("max_unknowns must be an integer of at least 2.")
        if time_domain is not None:
            try:
                interval = np.asarray(time_domain, dtype=np.float64)
            except (TypeError, ValueError):
                interval = np.asarray([], dtype=np.float64)
            if (interval.shape != (2,) or not np.all(np.isfinite(interval))
                    or interval[1] <= interval[0]):
                raise ValueError("time_domain must be a finite, increasing (start, stop) pair.")
            time_domain = (float(interval[0]), float(interval[1]))

        spatial_dims = geometry.dim
        dims = spatial_dims + int(time_domain is not None)
        registered_meta = get_pde_meta(pde) if pde else None
        # Let unregistered residual classes describe their own numerical
        # profile without requiring a PDE_REGISTRY entry.
        meta = {}
        for key, default in (
            ("order", 2), ("stiff", False), ("nonlinear", False),
            ("spectral_bias", "mixed"),
        ):
            value = getattr(residual, key, None)
            if value is None and registered_meta is not None:
                value = getattr(registered_meta, key, default)
            meta[key] = default if value is None else value
        variable_budget = min(int(max_unknowns), 384 // int(n_output))
        max_centers = variable_budget // int(n_output)
        if max_centers < 2:
            raise ValueError(
                f"n_output={n_output} exceeds the dense autosolver budget; "
                "at least two RBF centers are required. Increase max_unknowns or use a sparse discretization."
            )

        base = max(8, min(64, int(np.ceil(48.0 / dims))))
        reasons = [f"{dims}D space-time problem"]
        spectral = str(meta["spectral_bias"]).lower()
        multipliers = {"low": 0.90, "mixed": 1.05, "high": 1.25}
        base = int(np.ceil(base * multipliers.get(spectral, 1.0)))
        if meta["nonlinear"]:
            base = int(np.ceil(base * 1.10))
            reasons.append("nonlinear PDE allowance")
        if meta["stiff"]:
            base = int(np.ceil(base * 0.90))
            reasons.append("stiff-PDE conditioning allowance")
        order = int(meta["order"])
        if order >= 4:
            base = int(np.ceil(base * 0.75))
            reasons.append(f"order-{order} derivative conditioning allowance")
        if registered_meta is not None or any(hasattr(residual, key) for key in ("order", "stiff", "nonlinear", "spectral_bias")):
            source = "registered" if registered_meta is not None else "residual-provided"
            reasons.append(f"{source} {spectral}-spectrum PDE metadata")

        # Surface sample demand grows more slowly than volume demand. For a
        # space-time solve only spatial faces receive boundary conditions.
        default_interior = max(8, min(base, max(8, int(max_centers * 0.65))))
        if has_boundary_conditions:
            surface_estimate = int(np.ceil(default_interior ** ((spatial_dims - 1) / spatial_dims)))
            default_boundary = max(4, surface_estimate)
        else:
            default_boundary = 0
        if has_boundary_conditions and n_boundary is None:
            if max_centers < 3:
                raise ValueError(
                    "The dense autosolver budget needs at least two interior centers "
                    "and one boundary center when boundary conditions are supplied. "
                    "Increase max_unknowns or reduce n_output."
                )
            default_boundary = min(default_boundary, max_centers - 2)
            default_interior = min(default_interior, max_centers - default_boundary)
        if default_interior + default_boundary > max_centers:
            default_interior = max(2, max_centers - default_boundary)
            default_boundary = min(default_boundary, max_centers - default_interior)
        if not has_boundary_conditions and default_interior > max_centers:
            default_interior = max_centers
        if n_collocation is not None:
            if isinstance(n_collocation, (bool, np.bool_)) or int(n_collocation) != n_collocation:
                raise ValueError("n_collocation must be an integer.")
            reasons.append("user-specified interior point count")
        if n_boundary is not None:
            if isinstance(n_boundary, (bool, np.bool_)) or int(n_boundary) != n_boundary:
                raise ValueError("n_boundary must be an integer.")
            reasons.append("user-specified boundary point count")
        n_interior = int(n_collocation if n_collocation is not None else default_interior)
        n_bnd = int(n_boundary if n_boundary is not None else default_boundary)
        if n_interior < 1 or n_bnd < 0:
            raise ValueError("n_collocation must be positive and n_boundary non-negative.")
        if n_bnd and not has_boundary_conditions:
            raise ValueError("n_boundary requires at least one boundary condition.")
        if n_interior + n_bnd > max_centers:
            raise ValueError(
                f"Requested {n_interior + n_bnd} collocation/boundary centers, "
                f"but the current output/unknown budget allows {max_centers}. "
                "Increase max_unknowns or lower the requested point counts."
            )
        lengths = [float(hi - lo) for lo, hi in geometry.bounds]
        if any(not np.isfinite(length) or length <= 0 for length in lengths):
            raise ValueError("Geometry bounds must have finite, positive lengths.")
        if time_domain is not None:
            lengths.append(float(time_domain[1] - time_domain[0]))
        coordinate_scales = tuple(lengths)
        # All coordinates are normalized to their domain spans before the RBF
        # distance is evaluated. The spacing-based width therefore behaves
        # consistently on anisotropic domains and space-time problems.
        epsilon = float(rbf_epsilon) if rbf_epsilon is not None else (
            float(np.sqrt(dims) / max(3.0, (n_interior + n_bnd) ** (1.0 / dims)))
        )
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("rbf_epsilon must be positive.")
        ridge = float(rbf_ridge) if rbf_ridge is not None else 1e-8
        if rbf_epsilon is not None:
            reasons.append("user-specified RBF width")
        return AutoSolverConfig(
            n_collocation=n_interior,
            n_boundary=n_bnd,
            rbf_epsilon=epsilon,
            rbf_ridge=ridge,
            max_nfev=int(max_nfev),
            tolerance=float(tolerance),
            max_unknowns=variable_budget,
            coordinate_scales=coordinate_scales,
            residual_balance=bool(residual_balance),
            adaptive_ridge=rbf_ridge is None,
            selection_reason="; ".join(reasons),
        )


def _as_points(value: Any, spatial_dims: int, time_domain, *, rng, append_time: bool = True):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    valid_widths = {spatial_dims, spatial_dims + 1} if time_domain is not None else {spatial_dims}
    if points.ndim != 2 or points.shape[1] not in valid_widths:
        raise ValueError(f"Points must have {sorted(valid_widths)} columns, got {points.shape}.")
    if points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Points must be non-empty and contain only finite coordinates.")
    if time_domain is not None and points.shape[1] == spatial_dims + 1:
        if np.any(points[:, -1] < time_domain[0]) or np.any(points[:, -1] > time_domain[1]):
            raise ValueError("Space-time points must lie within time_domain.")
    if time_domain is not None and append_time and points.shape[1] == spatial_dims:
        t0, t1 = time_domain
        points = np.concatenate([points, rng.uniform(t0, t1, (len(points), 1))], axis=1)
    return points


def _initial_points(value: Any, spatial_dims: int, time_domain):
    """Put initial-condition samples on the initial time slice when omitted."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    valid_widths = {spatial_dims, spatial_dims + 1} if time_domain is not None else {spatial_dims}
    if points.ndim != 2 or points.shape[1] not in valid_widths:
        raise ValueError(f"Initial-condition points must have {sorted(valid_widths)} columns, got {points.shape}.")
    if points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Initial-condition points must be non-empty and finite.")
    if time_domain is not None and points.shape[1] == spatial_dims + 1:
        if np.any(points[:, -1] < time_domain[0]) or np.any(points[:, -1] > time_domain[1]):
            raise ValueError("Initial-condition points must lie within time_domain.")
        if not np.allclose(points[:, -1], time_domain[0]):
            raise ValueError("Initial-condition points with an explicit time column must lie at time_domain[0].")
    if time_domain is not None and points.shape[1] == spatial_dims:
        t0 = float(time_domain[0])
        points = np.concatenate([points, np.full((len(points), 1), t0)], axis=1)
    return points


def _target_array(value: Any, n: int, n_output: int, backend) -> np.ndarray:
    target = np.asarray(backend.to_numpy(value) if not isinstance(value, np.ndarray) else value)
    if target.ndim == 0:
        target = np.full((n, n_output), target.item())
    elif target.ndim == 1:
        if target.size == n:
            target = target[:, None]
        elif target.size == n_output:
            target = np.broadcast_to(target[None, :], (n, n_output))
    if target.shape[0] != n or target.shape[1] not in (1, n_output):
        raise ValueError(f"Constraint target must broadcast to ({n}, {n_output}), got {target.shape}.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Constraint targets must contain only finite values.")
    return np.broadcast_to(target, (n, n_output)).astype(np.float64, copy=False)


def _deduplicate(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return points
    scale = max(float(np.max(np.abs(points))), 1.0)
    return np.unique(np.round(points / scale, decimals=12), axis=0) * scale


class AutoSolver:
    """Automatically solve a supplied PDE residual by RBF collocation.

    Unlike a PINN, the unknowns are the values at a finite set of collocation
    centers. A smooth Gaussian RBF interpolant defines ``model_fn`` between
    centers, and SciPy solves the strong-form residual and supplied
    constraints as one classical nonlinear algebraic system.
    """

    def __init__(
        self,
        residual: Any = None,
        geometry: Optional[Geometry] = None,
        *,
        backend: Any = "torch",
        pde: Optional[str] = None,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        n_output: Optional[int] = None,
        time_domain: Optional[Tuple[float, float]] = None,
        boundary_conditions: Optional[BoundaryConditionSet] = None,
        ic_points: Any = None,
        ic_values: Any = None,
        data_points: Any = None,
        data_values: Any = None,
        n_collocation: Optional[int] = None,
        n_boundary: Optional[int] = None,
        rbf_epsilon: Optional[float] = None,
        rbf_ridge: Optional[float] = None,
        max_nfev: int = 200,
        tolerance: float = 1e-7,
        max_unknowns: int = 192,
        residual_balance: bool = True,
        initial_values: Any = 0.0,
        evaluation_points: Optional[np.ndarray] = None,
        n_evaluation: int = 1024,
        eval_time: Optional[float] = None,
        diff_mode: str = "reverse",
        seed: Optional[int] = 0,
        require_convergence: bool = True,
        optimizer: Optional[AutoSolverOptimizer] = None,
    ) -> None:
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a physai.geometry.Geometry.")
        if time_domain is not None:
            try:
                interval = np.asarray(time_domain, dtype=np.float64)
            except (TypeError, ValueError):
                interval = np.asarray([], dtype=np.float64)
            if (interval.shape != (2,) or not np.all(np.isfinite(interval))
                    or interval[1] <= interval[0]):
                raise ValueError("time_domain must be a finite, increasing (start, stop) pair.")
            time_domain = (float(interval[0]), float(interval[1]))
        if (ic_points is None) != (ic_values is None):
            raise ValueError("ic_points and ic_values must be supplied together.")
        if (data_points is None) != (data_values is None):
            raise ValueError("data_points and data_values must be supplied together.")
        if boundary_conditions is not None and boundary_conditions.geometry.dim != geometry.dim:
            raise ValueError("boundary_conditions geometry dimension must match geometry.")

        if isinstance(backend, str):
            backend = get_backend(backend)
        if isinstance(residual, str):
            pde = residual
            residual = None
        if residual is None:
            if not pde:
                raise ValueError("Supply a residual callable or a registered `pde` name.")
            from physai.core.pde_residual import build_residual
            residual = build_residual(pde, backend, **(residual_kwargs or {}))
        elif isinstance(residual, type):
            residual = residual(backend, **(residual_kwargs or {}))
        if not callable(residual):
            raise TypeError("residual must implement residual(model_fn, points).")

        if n_output is None:
            metadata = get_pde_meta(pde) if pde else None
            n_output = metadata.n_components if metadata is not None else int(
                getattr(residual, "n_components", getattr(residual, "n_output", 1))
            )
        if isinstance(n_output, (bool, np.bool_)) or int(n_output) != n_output or n_output < 1:
            raise ValueError("n_output must be a positive integer.")
        if isinstance(n_evaluation, (bool, np.bool_)) or int(n_evaluation) != n_evaluation or n_evaluation < 1:
            raise ValueError("n_evaluation must be a positive integer.")
        if eval_time is not None:
            if not np.isfinite(eval_time):
                raise ValueError("eval_time must be finite.")
            if time_domain is None:
                raise ValueError("eval_time requires time_domain.")
            if not (time_domain[0] <= eval_time <= time_domain[1]):
                raise ValueError("eval_time must lie within time_domain.")
        self.residual = residual
        self.geometry = geometry
        self.backend = backend
        self.pde = pde
        self.n_output = int(n_output)
        self.time_domain = time_domain
        self.boundary_conditions = boundary_conditions
        self.ic_points = ic_points
        self.ic_values = ic_values
        self.data_points = data_points
        self.data_values = data_values
        self.initial_values = initial_values
        self.evaluation_points = evaluation_points
        self.n_evaluation = int(n_evaluation)
        self.eval_time = eval_time
        self.diff_mode = diff_mode
        self.seed = seed
        self.require_convergence = require_convergence
        if optimizer is not None and not isinstance(optimizer, AutoSolverOptimizer):
            raise TypeError("optimizer must be an AutoSolverOptimizer instance or None.")
        self.optimizer = optimizer or AutoSolverOptimizer()
        self.config = self.optimizer.analyse(
            geometry,
            n_output=self.n_output,
            pde=pde,
            residual=residual,
            time_domain=time_domain,
            has_boundary_conditions=bool(
                boundary_conditions is not None and boundary_conditions.conditions
            ),
            n_collocation=n_collocation,
            n_boundary=n_boundary,
            rbf_epsilon=rbf_epsilon,
            rbf_ridge=rbf_ridge,
            max_nfev=max_nfev,
            tolerance=tolerance,
            max_unknowns=max_unknowns,
            residual_balance=residual_balance,
        )

    def solve(self) -> Dict[str, Any]:
        backend = self.backend
        rng = np.random.default_rng(self.seed)
        dims = self.geometry.dim
        cfg = self.config
        interior_spatial = self.geometry.sample_interior(cfg.n_collocation, rng=rng).astype(np.float64)
        if len(interior_spatial) != cfg.n_collocation:
            raise RuntimeError("Geometry could not supply enough interior collocation points.")
        interior_points = _as_points(interior_spatial, dims, self.time_domain, rng=rng)

        boundary_spatial = (
            self.geometry.sample_boundary(cfg.n_boundary, rng=rng).astype(np.float64)
            if cfg.n_boundary else np.empty((0, dims), dtype=np.float64)
        )
        if cfg.n_boundary and len(boundary_spatial) != cfg.n_boundary:
            raise RuntimeError(
                f"Geometry supplied {len(boundary_spatial)} of {cfg.n_boundary} requested boundary points; "
                "autosolve cannot enforce its boundary conditions reliably."
            )
        constraint_blocks = []
        centers = [interior_points]
        if self.ic_points is not None:
            ic_pts = _initial_points(self.ic_points, dims, self.time_domain)
            centers.append(ic_pts)
            constraint_blocks.append(("dirichlet", ic_pts, self.ic_values, None))
        if self.data_points is not None:
            data_pts = _as_points(self.data_points, dims, self.time_domain, rng=rng)
            centers.append(data_pts)
            constraint_blocks.append(("dirichlet", data_pts, self.data_values, None))

        if self.boundary_conditions is not None and len(boundary_spatial):
            assigned = self.boundary_conditions.assign(boundary_spatial)
            conditions = {bc.name: bc for bc in self.boundary_conditions.conditions}
            unassigned = [
                name for name, spatial_pts in assigned.items()
                if name in conditions and not len(spatial_pts)
            ]
            if unassigned:
                raise RuntimeError(
                    "Boundary sampling did not hit condition region(s) "
                    f"{unassigned}; increase n_boundary or broaden the regions."
                )
            for name, spatial_pts in assigned.items():
                bc = conditions.get(name)
                if bc is None or not len(spatial_pts):
                    continue
                time_values = None
                if self.time_domain is not None:
                    t0, t1 = self.time_domain
                    time_values = rng.uniform(t0, t1, size=(len(spatial_pts), 1))
                pts = spatial_pts if time_values is None else np.concatenate([spatial_pts, time_values], axis=1)
                target = None if bc.kind == "periodic" else bc.value(spatial_pts)
                pair = None
                if bc.kind == "periodic":
                    pair_spatial = bc.pair_map(spatial_pts)
                    pair_spatial = np.asarray(pair_spatial, dtype=np.float64)
                    if pair_spatial.shape != spatial_pts.shape or not np.all(np.isfinite(pair_spatial)):
                        raise ValueError(
                            f"Periodic pair_map for {bc.name!r} must return finite points with shape {spatial_pts.shape}."
                        )
                    pair = pair_spatial if time_values is None else np.concatenate([pair_spatial, time_values], axis=1)
                centers.extend([pts] + ([pair] if pair is not None else []))
                constraint_blocks.append((bc.kind, pts, target, (bc, pair)))

        centers_np = _deduplicate(np.concatenate(centers, axis=0))
        if len(centers_np) * self.n_output > cfg.max_unknowns:
            raise ValueError(
                f"AutoSolver selected {len(centers_np) * self.n_output} unknowns, "
                f"above its dense RBF limit ({cfg.max_unknowns}). Lower collocation/BC "
                "counts or use solve_discrete_geometry with a sparse discretization."
            )
        if len(centers_np) < 2:
            raise ValueError("RBF collocation needs at least two distinct points.")

        coordinate_scales = np.asarray(cfg.coordinate_scales, dtype=np.float64)
        centers_scaled = centers_np / coordinate_scales[None, :]
        distances, _ = cKDTree(centers_scaled).query(centers_scaled, k=2)
        positive = distances[:, 1][distances[:, 1] > 0]
        epsilon = cfg.rbf_epsilon if not len(positive) else max(
            cfg.rbf_epsilon, float(np.median(positive))
        )
        diffs = centers_scaled[:, None, :] - centers_scaled[None, :, :]
        kernel_matrix = np.exp(-np.sum(diffs * diffs, axis=-1) / (epsilon * epsilon))
        effective_ridge = cfg.rbf_ridge
        if cfg.adaptive_ridge:
            eigenvalues = np.linalg.eigvalsh(kernel_matrix)
            largest = max(float(eigenvalues[-1]), 1.0)
            smallest = float(eigenvalues[0])
            # The RBF solve is eventually evaluated using the selected tensor
            # backend (usually float32), so target a safer condition number
            # than a double-precision-only linear solve would permit.
            condition_limit = 1e8
            needed = max(0.0, (largest - condition_limit * smallest) / (condition_limit - 1.0))
            effective_ridge = max(effective_ridge, needed)
        kernel_matrix.flat[::len(centers_np) + 1] += effective_ridge
        try:
            value_to_coeff = dense_solve(kernel_matrix, np.eye(len(centers_np)), assume_a="pos")
        except Exception:
            value_to_coeff = np.linalg.pinv(kernel_matrix, rcond=1e-12)
        centers_t = backend.tensor(centers_scaled.astype(np.float32))
        coordinate_scales_t = backend.tensor(coordinate_scales.astype(np.float32))
        value_to_coeff_t = backend.tensor(value_to_coeff.astype(np.float32))
        collocation_t = backend.tensor(interior_points.astype(np.float32))
        boundary_tensors = []
        for kind, pts, target, extra in constraint_blocks:
            boundary_tensors.append((kind, backend.tensor(pts.astype(np.float32)), target, extra))

        def _make_model(values):
            coefficients = backend.matmul(value_to_coeff_t, values)

            def model_fn(query):
                normalized_query = query / coordinate_scales_t
                delta = normalized_query[:, None, :] - centers_t[None, :, :]
                r2 = backend.sum(backend.square(delta), axis=-1)
                kernel = backend.exp(-r2 / (epsilon * epsilon))
                return backend.matmul(kernel, coefficients)
            return model_fn

        def _evaluate_unknowns(flat_values, *, balanced=None):
            if balanced is None:
                balanced = cfg.residual_balance
            nodal_values = backend.tensor(np.asarray(flat_values, dtype=np.float32).reshape(-1, self.n_output))
            model_fn = _make_model(nodal_values)
            blocks = []

            def _append_block(value):
                block = np.asarray(value, dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(block)):
                    raise FloatingPointError("PDE/constraint residual produced NaN or infinite values.")
                if balanced and block.size:
                    block = block / np.sqrt(block.size)
                blocks.append(block)

            pde_residual = backend.to_numpy(self.residual(model_fn, collocation_t))
            _append_block(pde_residual)
            for kind, pts_t, target, extra in boundary_tensors:
                prediction = model_fn(pts_t)
                if kind == "periodic":
                    _bc, pair = extra
                    _append_block(backend.to_numpy(prediction - model_fn(backend.tensor(pair.astype(np.float32)))))
                elif kind == "dirichlet":
                    target_arr = _target_array(target, len(pts_t), self.n_output, backend)
                    _append_block(np.asarray(backend.to_numpy(prediction), dtype=np.float64).reshape(-1) - target_arr.reshape(-1))
                elif kind in ("neumann", "robin"):
                    bc, _pair = extra
                    normal = self.geometry.normal(np.asarray(backend.to_numpy(pts_t))[:, :dims])
                    derivatives = []
                    for output_col in range(self.n_output):
                        grad_fn = backend.grad(
                            lambda x, _col=output_col: backend.sum(model_fn(x)[..., _col]),
                            mode=self.diff_mode,
                        )
                        derivatives.append(grad_fn(pts_t)[..., :dims])
                    gradient = backend.stack(derivatives, axis=-2)
                    normal_t = backend.tensor(normal.astype(np.float32))[:, None, :]
                    normal_derivative = backend.sum(gradient * normal_t, axis=-1)
                    if kind == "robin":
                        prediction = bc.coeff_a * prediction + bc.coeff_b * normal_derivative
                    else:
                        prediction = normal_derivative
                    target_arr = _target_array(target, len(pts_t), self.n_output, backend)
                    _append_block(np.asarray(backend.to_numpy(prediction), dtype=np.float64).reshape(-1) - target_arr.reshape(-1))
            return np.concatenate(blocks)

        initial = np.asarray(self.initial_values, dtype=np.float64)
        if initial.ndim == 0:
            initial = np.full((len(centers_np), self.n_output), initial.item())
        elif initial.shape != (len(centers_np), self.n_output):
            raise ValueError(f"initial_values must be scalar or shape ({len(centers_np)}, {self.n_output}).")

        fit = least_squares(
            _evaluate_unknowns,
            initial.reshape(-1),
            method="trf",
            max_nfev=cfg.max_nfev,
            ftol=cfg.tolerance,
            xtol=cfg.tolerance,
            gtol=cfg.tolerance,
            x_scale="jac",
        )
        final_residual = _evaluate_unknowns(fit.x, balanced=False)
        residual_norm = float(np.linalg.norm(final_residual, ord=np.inf))
        if self.require_convergence and (not fit.success or residual_norm > max(cfg.tolerance * 10.0, 1e-6)):
            raise RuntimeError(
                f"AutoSolver did not converge: {fit.message}; "
                f"max residual={residual_norm:.3e}. Increase resolution or adjust constraints."
            )

        values = backend.tensor(fit.x.astype(np.float32).reshape(-1, self.n_output))
        model_fn = _make_model(values)
        if self.evaluation_points is None:
            eval_spatial = self.geometry.sample_interior(self.n_evaluation, rng=rng).astype(np.float64)
            if len(eval_spatial) < 1:
                raise RuntimeError("Geometry could not supply evaluation points.")
            eval_points = _as_points(eval_spatial, dims, self.time_domain, rng=rng)
            if self.time_domain is not None and self.eval_time is not None:
                eval_points[:, -1] = self.eval_time
        else:
            eval_points = _as_points(self.evaluation_points, dims, self.time_domain, rng=rng)
        pred = np.asarray(backend.to_numpy(model_fn(backend.tensor(eval_points.astype(np.float32)))))
        if pred.ndim == 1:
            pred = pred[:, None]
        declared_names = getattr(self.residual, "fields", None)
        if declared_names is not None and len(declared_names) == self.n_output:
            output_names = list(declared_names)
        else:
            output_names = ["u"] if self.n_output == 1 else [f"u{i}" for i in range(self.n_output)]
        return {
            "coordinates": eval_points,
            "values": {name: pred[:, i] for i, name in enumerate(output_names)},
            "model_fn": model_fn,
            "centers": centers_np,
            "nodal_values": fit.x.reshape(-1, self.n_output),
            "optimizer_result": fit,
            "config": cfg,
            "optimizer": self.optimizer,
            "residual_norm": residual_norm,
            "effective_rbf_epsilon": epsilon,
            "effective_rbf_ridge": effective_ridge,
        }


def autosolve(
    residual: Any = None,
    geometry: Optional[Geometry] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience pipeline for registered or caller-supplied PDE residuals."""
    return AutoSolver(residual, geometry, **kwargs).solve()


__all__ = ["AutoSolverConfig", "AutoSolverOptimizer", "AutoSolver", "autosolve"]
