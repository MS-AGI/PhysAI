"""
physai/utils.py

Cross-cutting utility functions for PhysAI.

Sections
--------
1.  Sampling strategies       — Latin Hypercube, Sobol, uniform, adaptive
2.  Normalisation helpers     — input/output min-max & z-score scalers
3.  Metrics                   — L2 relative error, L∞, RMSE, R²
4.  Domain utilities          — boundary extraction, distance functions
5.  Reproducibility           — global seed setting across all backends
6.  Logging                   — structured JSON logger + TensorBoard shim
7.  Dtype / device helpers    — safe casting, device introspection
8.  Miscellaneous             — timer, progress bar, flatten/unflatten dicts
9.  Arbitrary geometry        — re-exported from ``geometry.py``: SDF
                                 primitives, CSG (union/intersection/
                                 difference), user-defined shapes written
                                 directly in code, mesh files uploaded by
                                 the user (.stl/.obj/.ply/.off), and a
                                 boundary-condition system (Dirichlet /
                                 Neumann / Robin / periodic) that works on
                                 any of the above. See geometry.py for the
                                 full implementation and docstrings — kept
                                 in its own module since it's a large,
                                 self-contained subsystem, and re-exported
                                 here so `from physai.utils import *`
                                 keeps working the way it always has.
"""
from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import (
    Geometry,
    box, ball, ellipsoid, cylinder, torus, capsule, half_space, polygon,
    translate, scale, rotate2d, rotate3d,
    union, intersection, difference, invert, smooth_union, smooth_difference,
    custom_geometry, geometry_from_file,
    BoundaryCondition, BoundaryConditionSet, face_region, everywhere,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger("physai")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[PhysAI %(levelname)s] %(message)s"))
    _logger.addHandler(_h)


# ============================================================================
# 1. Sampling
# ============================================================================

def sample_uniform(
    bounds: List[Tuple[float, float]],
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw ``n`` points uniformly from the hyper-rectangle defined by ``bounds``.

    Parameters
    ----------
    bounds : list of (lo, hi) per dimension
    n      : number of points
    rng    : numpy Generator (created from global seed if None)

    Returns
    -------
    np.ndarray of shape (n, d)
    """
    rng = rng or np.random.default_rng()
    d   = len(bounds)
    pts = np.empty((n, d), dtype=np.float32)
    for i, (lo, hi) in enumerate(bounds):
        pts[:, i] = rng.uniform(lo, hi, n)
    return pts


def sample_lhs(
    bounds: List[Tuple[float, float]],
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Latin Hypercube Sampling — better space-filling than pure random.

    Each dimension is divided into ``n`` equal intervals; exactly one
    sample falls in each interval per dimension, chosen uniformly within it.
    """
    rng = rng or np.random.default_rng()
    d   = len(bounds)
    pts = np.empty((n, d), dtype=np.float32)
    for i, (lo, hi) in enumerate(bounds):
        cuts   = np.linspace(lo, hi, n + 1)
        lo_v   = cuts[:-1]
        hi_v   = cuts[1:]
        sample = rng.uniform(lo_v, hi_v)
        pts[:, i] = rng.permutation(sample)
    return pts


def sample_sobol(
    bounds: List[Tuple[float, float]],
    n: int,
    skip: int = 1,
) -> np.ndarray:
    """
    Quasi-random Sobol sequence sampling.
    Requires ``scipy >= 1.7``.

    Parameters
    ----------
    bounds : list of (lo, hi)
    n      : number of points  (will be rounded up to next power of 2)
    skip   : number of initial Sobol points to skip (reduces correlation artefacts)
    """
    try:
        from scipy.stats.qmc import Sobol
    except ImportError:
        _logger.warning(
            "scipy.stats.qmc not available; falling back to LHS sampling."
        )
        return sample_lhs(bounds, n)

    d       = len(bounds)
    n_pow2  = int(2 ** math.ceil(math.log2(max(n + skip, 2))))
    sampler = Sobol(d=d, scramble=True)
    raw     = sampler.random_base2(m=int(math.log2(n_pow2)))[skip : skip + n]

    pts = np.empty((len(raw), d), dtype=np.float32)
    for i, (lo, hi) in enumerate(bounds):
        pts[:, i] = lo + (hi - lo) * raw[:, i]
    return pts


def sample_boundary(
    bounds: List[Tuple[float, float]],
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample ``n`` points on the boundary of a hyper-rectangle.
    Each point is placed on a randomly selected face; coordinates on
    non-selected dimensions are drawn uniformly from their range.
    """
    rng = rng or np.random.default_rng()
    d   = len(bounds)
    pts: List[np.ndarray] = []

    for _ in range(n):
        pt  = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float32)
        dim = int(rng.integers(0, d))
        lo, hi = bounds[dim]
        pt[dim] = lo if rng.random() < 0.5 else hi
        pts.append(pt)

    return np.stack(pts, axis=0)


def sample_initial_condition(
    spatial_bounds: List[Tuple[float, float]],
    t0: float,
    n: int,
    method: str = "lhs",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample ``n`` points on the initial-time hyperplane t = t0.
    The returned array has shape (n, d+1) with the last column = t0.
    """
    rng = rng or np.random.default_rng()
    fn  = {"lhs": sample_lhs, "uniform": sample_uniform, "sobol": sample_sobol}
    if method not in fn:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(fn)}.")

    if method == "sobol":
        spatial = fn[method](spatial_bounds, n)
    else:
        spatial = fn[method](spatial_bounds, n, rng=rng)

    t_col = np.full((len(spatial), 1), t0, dtype=np.float32)
    return np.concatenate([spatial, t_col], axis=1)


def sample_adaptive(
    model_fn: Callable[[np.ndarray], np.ndarray],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    bounds: List[Tuple[float, float]],
    n_add: int,
    n_candidates: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Residual-driven adaptive sampling (numpy-only version, no backend needed).

    Evaluates the residual at ``n_candidates`` random points and returns
    the ``n_add`` points with the highest residual magnitude.

    Useful for pre-training seeding or pure-numpy workflows.
    """
    rng  = rng or np.random.default_rng()
    cand = sample_uniform(bounds, n_candidates, rng=rng)
    res  = residual_fn(cand)
    if res.ndim > 1:
        mag = np.mean(res ** 2, axis=-1)
    else:
        mag = res ** 2
    idx = np.argsort(mag)[::-1][:n_add]
    return cand[idx]


# ============================================================================
# 2. Normalisation
# ============================================================================

@dataclass
class MinMaxScaler:
    """
    Per-feature min-max normalisation to [out_min, out_max].
    Fit on numpy arrays; transform returns numpy arrays.
    """
    out_min: float = 0.0
    out_max: float = 1.0
    _data_min: Optional[np.ndarray] = field(default=None, repr=False)
    _data_max: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self._data_min = X.min(axis=0)
        self._data_max = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._data_min is None:
            raise RuntimeError("Call fit() before transform().")
        scale = (self.out_max - self.out_min) / (
            self._data_max - self._data_min + 1e-12
        )
        return self.out_min + (X - self._data_min) * scale

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._data_min is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        scale = (self._data_max - self._data_min) / (
            self.out_max - self.out_min + 1e-12
        )
        return self._data_min + (X_scaled - self.out_min) * scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class ZScoreScaler:
    """
    Per-feature z-score standardisation: (X - μ) / σ.
    """
    _mean: Optional[np.ndarray] = field(default=None, repr=False)
    _std:  Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self._mean) / self._std

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return X_scaled * self._std + self._mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# 3. Metrics
# ============================================================================

def relative_l2_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """
    Relative L² error: ||u_pred - u_ref||₂ / (||u_ref||₂ + ε).
    """
    num = float(np.sqrt(np.sum((prediction - reference) ** 2)))
    den = float(np.sqrt(np.sum(reference ** 2))) + epsilon
    return num / den


def relative_linf_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Relative L∞ error."""
    return float(np.max(np.abs(prediction - reference))) / (
        float(np.max(np.abs(reference))) + epsilon
    )


def rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def r_squared(prediction: np.ndarray, reference: np.ndarray) -> float:
    """
    Coefficient of determination R² = 1 - SS_res / SS_tot.
    R² = 1 → perfect fit; R² < 0 → worse than the mean predictor.
    """
    ss_res = float(np.sum((reference - prediction) ** 2))
    ss_tot = float(np.sum((reference - reference.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def compute_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """Return all standard metrics as a dict."""
    return {
        "rel_l2":   relative_l2_error(prediction, reference),
        "rel_linf": relative_linf_error(prediction, reference),
        "rmse":     rmse(prediction, reference),
        "r2":       r_squared(prediction, reference),
    }


# ============================================================================
# 4. Domain utilities
# ============================================================================

def rectangular_distance_fn(
    bounds: List[Tuple[float, float]],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a smooth distance-to-boundary function for a hyper-rectangle.

    D(x) = ∏_i (x_i - lo_i)(hi_i - x_i)   (product of linear factors)

    D = 0 on ∂Ω, D > 0 in Ω interior.  Used for hard Dirichlet constraints.
    """
    def _dist(x: np.ndarray) -> np.ndarray:
        d = np.ones(x.shape[0], dtype=np.float32)
        for i, (lo, hi) in enumerate(bounds):
            d *= (x[:, i] - lo) * (hi - x[:, i])
        return d
    return _dist


def spherical_distance_fn(
    center: Sequence[float],
    radius: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Smooth distance function for a d-dimensional ball:
    D(x) = R² - ||x - c||²   (zero on sphere, positive inside).
    """
    c = np.array(center, dtype=np.float32)
    def _dist(x: np.ndarray) -> np.ndarray:
        return radius ** 2 - np.sum((x - c) ** 2, axis=-1)
    return _dist


def extract_boundary_points(
    interior: np.ndarray,
    bounds: List[Tuple[float, float]],
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Extract points from ``interior`` that lie within ``tol`` of any face.
    """
    mask = np.zeros(len(interior), dtype=bool)
    for i, (lo, hi) in enumerate(bounds):
        mask |= (np.abs(interior[:, i] - lo) < tol)
        mask |= (np.abs(interior[:, i] - hi) < tol)
    return interior[mask]


# ============================================================================
# 5. Reproducibility
# ============================================================================

def set_global_seed(seed: int, backends: Optional[List[str]] = None) -> None:
    """
    Set random seeds for reproducibility across all installed backends.

    Parameters
    ----------
    seed     : integer seed value
    backends : list of backend names to seed; defaults to all available
    """
    np.random.seed(seed)

    available = backends or ["torch", "jax", "tensorflow"]

    if "torch" in available:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
        except ImportError:
            pass

    if "jax" in available:
        try:
            import jax
            # JAX uses explicit keys; we just ensure determinism flag is set
            jax.config.update("jax_default_prng_impl", "threefry2x32")
        except ImportError:
            pass

    if "tensorflow" in available:
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

    _logger.info(f"Global seed set to {seed}.")


# ============================================================================
# 6. Logging
# ============================================================================

class JSONLogger:
    """
    Append-mode JSON-Lines logger. Each log call appends a JSON record
    to ``path``. Safe for long runs — does not accumulate in memory.

    Example
    -------
    >>> logger = JSONLogger("run/metrics.jsonl")
    >>> logger.log(step=100, loss=1.2e-3, lr=5e-4)
    """

    def __init__(self, path: str, overwrite: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        self._fh = self._path.open(mode, buffering=1, encoding="utf-8")   # line-buffered

    def log(self, **kwargs: Any) -> None:
        kwargs["_ts"] = time.time()
        self._fh.write(json.dumps(kwargs) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all records from the file."""
        records = []
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


class TensorBoardLogger:
    """
    Thin TensorBoard (SummaryWriter) shim.
    Falls back to a no-op if tensorboard is not installed.
    """

    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
            self._available = True
        except ImportError:
            self._writer   = None
            self._available = False
            _logger.warning(
                "tensorboard not available; TensorBoardLogger is a no-op."
            )

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._available:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        if self._available:
            self._writer.add_scalars(tag, values, step)

    def close(self) -> None:
        if self._available:
            self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ============================================================================
# 7. Dtype / device helpers
# ============================================================================

def to_float32(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 in-place-free manner."""
    return arr.astype(np.float32, copy=False)


def to_float64(arr: np.ndarray) -> np.ndarray:
    """Cast to float64 in-place-free manner."""
    return arr.astype(np.float64, copy=False)


def detect_devices() -> Dict[str, List[str]]:
    """
    Detect available compute devices across installed backends.

    Returns
    -------
    dict with keys "torch", "jax", "tensorflow", each mapping to a
    list of device strings.
    """
    result: Dict[str, List[str]] = {}

    try:
        import torch
        devs = ["cpu"]
        if torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.append("mps")
        result["torch"] = devs
    except ImportError:
        result["torch"] = []

    try:
        import jax
        result["jax"] = [str(d) for d in jax.devices()]
    except ImportError:
        result["jax"] = []

    try:
        import tensorflow as tf
        result["tensorflow"] = [d.name for d in tf.config.list_physical_devices()]
    except ImportError:
        result["tensorflow"] = []

    return result


def get_dtype_string(backend_name: str, dtype: Any) -> str:
    """Return a human-readable dtype string regardless of backend type."""
    dtype_str = str(dtype)
    if "float32" in dtype_str or dtype_str == "<class 'torch.float32'>":
        return "float32"
    if "float64" in dtype_str or dtype_str == "<class 'torch.float64'>":
        return "float64"
    if "float16" in dtype_str:
        return "float16"
    return dtype_str


# ============================================================================
# 8. Miscellaneous
# ============================================================================

@contextmanager
def timer(label: str = "", logger: Optional[logging.Logger] = None) -> Generator:
    """
    Context manager that measures and logs wall time.

    >>> with timer("forward pass"):
    ...     y = model(x)
    """
    t0  = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        msg     = f"{label + ': ' if label else ''}{elapsed:.4f}s"
        (logger or _logger).info(msg)


class ProgressBar:
    """
    Minimal ASCII progress bar (no dependencies).

    >>> pb = ProgressBar(total=1000, width=40)
    >>> for i in range(1000):
    ...     pb.update(i + 1, loss=f"{0.1/(i+1):.3e}")
    """

    def __init__(
        self,
        total: int,
        width: int = 50,
        prefix: str = "Training",
    ) -> None:
        self.total  = total
        self.width  = width
        self.prefix = prefix
        self._t0    = time.perf_counter()

    def update(self, n: int, **kwargs: Any) -> None:
        frac    = min(n / max(self.total, 1), 1.0)
        filled  = int(self.width * frac)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.perf_counter() - self._t0
        eta_s   = (elapsed / frac - elapsed) if frac > 0 else 0.0
        extras  = "  ".join(f"{k}={v}" for k, v in kwargs.items())
        print(
            f"\r{self.prefix} |{bar}| {n}/{self.total}  "
            f"ETA {eta_s:.0f}s  {extras}",
            end="",
            flush=True,
        )
        if n >= self.total:
            print()


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dict.

    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """Inverse of ``flatten_dict``."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        node  = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def count_parameters_numpy(params: List[np.ndarray]) -> int:
    """Count the total number of scalar parameters in a list of arrays."""
    return sum(int(np.prod(p.shape)) for p in params)


def format_number(n: Union[int, float]) -> str:
    """Human-readable number: 1_234_567 → '1.23 M'."""
    if n < 1_000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1_000:.2f} K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f} M"
    return f"{n / 1_000_000_000:.2f} B"


def pairwise_distance(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of X (and Y).

    Returns
    -------
    np.ndarray of shape (len(X), len(Y or X))
    """
    Y = X if Y is None else Y
    # ||x - y||² = ||x||² + ||y||² - 2 x·y
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True)
    D2 = X2 + Y2.T - 2.0 * (X @ Y.T)
    return np.sqrt(np.maximum(D2, 0.0))


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    # Sampling
    "sample_uniform",
    "sample_lhs",
    "sample_sobol",
    "sample_boundary",
    "sample_initial_condition",
    "sample_adaptive",
    # Normalisation
    "MinMaxScaler",
    "ZScoreScaler",
    # Metrics
    "relative_l2_error",
    "relative_linf_error",
    "rmse",
    "r_squared",
    "compute_metrics",
    # Domain
    "rectangular_distance_fn",
    "spherical_distance_fn",
    "extract_boundary_points",
    # Reproducibility
    "set_global_seed",
    # Logging
    "JSONLogger",
    "TensorBoardLogger",
    # Dtype / device
    "to_float32",
    "to_float64",
    "detect_devices",
    "get_dtype_string",
    # Misc
    "timer",
    "ProgressBar",
    "flatten_dict",
    "unflatten_dict",
    "count_parameters_numpy",
    "format_number",
    "pairwise_distance",
    # Arbitrary geometry (see geometry.py)
    "Geometry",
    "box", "ball", "ellipsoid", "cylinder", "torus", "capsule",
    "half_space", "polygon",
    "translate", "scale", "rotate2d", "rotate3d",
    "union", "intersection", "difference", "invert",
    "smooth_union", "smooth_difference",
    "custom_geometry", "geometry_from_file",
    "BoundaryCondition", "BoundaryConditionSet",
    "face_region", "everywhere",
]