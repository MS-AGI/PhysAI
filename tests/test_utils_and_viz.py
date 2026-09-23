"""
tests/test_utils_and_viz.py

Combined coverage-focused tests for:
  - physai.utils
  - physai.visualization
  - physai.visualization_nd

Goal: exercise as many code paths (branches, optional args, fallbacks)
as possible in one file. Uses matplotlib's non-interactive "Agg" backend
so nothing tries to open a window during the run.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # noqa: E402  -- must happen before pyplot import

import matplotlib.pyplot as plt  # noqa: E402

from physai import utils as U  # noqa: E402
from physai import visualization as V  # noqa: E402
from physai import visualization_nd as VN  # noqa: E402


# ============================================================================
# Fixtures / helpers
# ============================================================================

@pytest.fixture(autouse=True)
def _close_all_figures():
    """Prevent matplotlib figures from piling up across tests."""
    yield
    plt.close("all")


class _History:
    """Minimal stand-in for a TrainingHistory object."""

    def __init__(self, n=50, with_val=True, terms=("pde", "bc")):
        self.steps = list(range(n))
        self.total_loss = list(np.linspace(1.0, 0.01, n))
        self.terms = {t: list(np.linspace(0.5, 0.001, n)) for t in terms}
        self.val_loss = list(np.linspace(1.1, 0.02, 5)) if with_val else []


# ============================================================================
# 1. utils.py -- Sampling
# ============================================================================

def test_sample_uniform_shape_and_bounds():
    bounds = [(0.0, 1.0), (-2.0, 2.0)]
    pts = U.sample_uniform(bounds, 200, rng=np.random.default_rng(0))
    assert pts.shape == (200, 2)
    assert (pts[:, 0] >= 0.0).all() and (pts[:, 0] <= 1.0).all()
    assert (pts[:, 1] >= -2.0).all() and (pts[:, 1] <= 2.0).all()


def test_sample_uniform_default_rng():
    pts = U.sample_uniform([(0.0, 1.0)], 5)
    assert pts.shape == (5, 1)


def test_sample_lhs_shape_and_bounds():
    bounds = [(0.0, 10.0), (5.0, 6.0), (-1.0, 1.0)]
    pts = U.sample_lhs(bounds, 32, rng=np.random.default_rng(1))
    assert pts.shape == (32, 3)
    for i, (lo, hi) in enumerate(bounds):
        assert (pts[:, i] >= lo).all() and (pts[:, i] <= hi).all()


def test_sample_sobol_fallback_without_scipy(monkeypatch):
    # Force the ImportError branch regardless of whether scipy is installed.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "scipy.stats.qmc" or name.startswith("scipy.stats.qmc"):
            raise ImportError("forced for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    pts = U.sample_sobol([(0.0, 1.0), (0.0, 1.0)], 16)
    assert pts.shape[1] == 2


def test_sample_sobol_with_scipy_if_available():
    scipy_qmc = pytest.importorskip("scipy.stats.qmc")
    pts = U.sample_sobol([(0.0, 1.0), (-1.0, 1.0)], 10, skip=2)
    assert pts.shape[1] == 2
    assert (pts[:, 0] >= 0.0).all() and (pts[:, 0] <= 1.0).all()


def test_sample_boundary_on_faces():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    pts = U.sample_boundary(bounds, 50, rng=np.random.default_rng(2))
    assert pts.shape == (50, 2)
    on_boundary = np.isclose(pts, 0.0) | np.isclose(pts, 1.0)
    assert on_boundary.any(axis=1).all()


@pytest.mark.parametrize("method", ["lhs", "uniform", "sobol"])
def test_sample_initial_condition_methods(method):
    if method == "sobol":
        pytest.importorskip("scipy.stats.qmc")
    pts = U.sample_initial_condition([(0.0, 1.0)], t0=0.5, n=8, method=method)
    assert pts.shape == (8, 2)
    assert np.allclose(pts[:, -1], 0.5)


def test_sample_initial_condition_invalid_method():
    with pytest.raises(ValueError):
        U.sample_initial_condition([(0.0, 1.0)], t0=0.0, n=4, method="bogus")


def test_sample_adaptive_picks_highest_residual():
    bounds = [(0.0, 1.0)]

    def residual_fn(x):
        return x[:, 0]  # residual grows with x

    picked = U.sample_adaptive(
        model_fn=lambda x: x,
        residual_fn=residual_fn,
        bounds=bounds,
        n_add=5,
        n_candidates=200,
        rng=np.random.default_rng(3),
    )
    assert picked.shape == (5, 1)
    # The picked points should generally sit in the upper range of [0, 1]
    assert picked.mean() > 0.5


def test_sample_adaptive_multi_output_residual():
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def residual_fn(x):
        return np.stack([x[:, 0], x[:, 1]], axis=-1)

    picked = U.sample_adaptive(
        model_fn=lambda x: x,
        residual_fn=residual_fn,
        bounds=bounds,
        n_add=3,
        n_candidates=100,
    )
    assert picked.shape == (3, 2)


# ============================================================================
# 2. utils.py -- Normalisation
# ============================================================================

def test_minmax_scaler_roundtrip():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler = U.MinMaxScaler(out_min=0.0, out_max=1.0)
    Xs = scaler.fit_transform(X)
    assert np.isclose(Xs.min(), 0.0)
    assert np.isclose(Xs.max(), 1.0)
    X_back = scaler.inverse_transform(Xs)
    assert np.allclose(X_back, X, atol=1e-4)


def test_minmax_scaler_errors_before_fit():
    scaler = U.MinMaxScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        scaler.inverse_transform(np.zeros((3, 2)))


def test_zscore_scaler_roundtrip():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
    scaler = U.ZScoreScaler()
    Xs = scaler.fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-6)
    X_back = scaler.inverse_transform(Xs)
    assert np.allclose(X_back, X, atol=1e-4)


def test_zscore_scaler_errors_before_fit():
    scaler = U.ZScoreScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        scaler.inverse_transform(np.zeros((3, 2)))


# ============================================================================
# 3. utils.py -- Metrics
# ============================================================================

def test_metrics_perfect_prediction():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = ref.copy()
    assert U.relative_l2_error(pred, ref) == pytest.approx(0.0, abs=1e-9)
    assert U.relative_linf_error(pred, ref) == pytest.approx(0.0, abs=1e-9)
    assert U.rmse(pred, ref) == pytest.approx(0.0, abs=1e-9)
    assert U.r_squared(pred, ref) == pytest.approx(1.0, abs=1e-9)


def test_metrics_imperfect_prediction():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = ref + 1.0
    assert U.relative_l2_error(pred, ref) > 0
    assert U.relative_linf_error(pred, ref) > 0
    assert U.rmse(pred, ref) == pytest.approx(1.0, abs=1e-6)
    assert U.r_squared(pred, ref) < 1.0


def test_compute_metrics_returns_all_keys():
    ref = np.linspace(0, 1, 20)
    pred = ref + np.random.default_rng(5).normal(scale=0.01, size=20)
    m = U.compute_metrics(pred, ref)
    assert set(m.keys()) == {"rel_l2", "rel_linf", "rmse", "r2"}
    assert all(isinstance(v, float) for v in m.values())


# ============================================================================
# 4. utils.py -- Domain utilities
# ============================================================================

def test_rectangular_distance_fn_zero_on_boundary():
    dist = U.rectangular_distance_fn([(0.0, 1.0), (0.0, 1.0)])
    boundary_pts = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
    d = dist(boundary_pts)
    assert np.allclose(d, 0.0, atol=1e-6)
    interior = dist(np.array([[0.5, 0.5]]))
    assert (interior > 0).all()


def test_spherical_distance_fn():
    dist = U.spherical_distance_fn(center=[0.0, 0.0], radius=1.0)
    center_val = dist(np.array([[0.0, 0.0]]))
    edge_val = dist(np.array([[1.0, 0.0]]))
    outside_val = dist(np.array([[2.0, 0.0]]))
    assert center_val[0] > 0
    assert np.isclose(edge_val[0], 0.0, atol=1e-6)
    assert outside_val[0] < 0


def test_extract_boundary_points():
    interior = np.array([
        [0.0, 0.5],   # on left face
        [0.5, 0.5],   # interior
        [1.0, 1.0],   # corner (on two faces)
        [0.3, 0.3],   # interior
    ])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    b = U.extract_boundary_points(interior, bounds)
    assert len(b) == 2
    assert any(np.allclose(row, [0.0, 0.5]) for row in b)
    assert any(np.allclose(row, [1.0, 1.0]) for row in b)


# ============================================================================
# 5. utils.py -- Reproducibility
# ============================================================================

def test_set_global_seed_numpy_only():
    U.set_global_seed(42, backends=[])
    a = np.random.rand(5)
    U.set_global_seed(42, backends=[])
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_set_global_seed_default_backends_no_crash():
    # backends=None -> tries torch/jax/tensorflow, all optional (ImportError
    # is swallowed internally), so this should never raise.
    U.set_global_seed(123)


# ============================================================================
# 6. utils.py -- Logging
# ============================================================================

def test_json_logger_write_and_read(tmp_path):
    log_path = tmp_path / "sub" / "metrics.jsonl"
    logger = U.JSONLogger(str(log_path))
    logger.log(step=1, loss=0.5)
    logger.log(step=2, loss=0.25)
    logger.close()

    records = U.JSONLogger(str(log_path)).read_all()
    assert len(records) == 2
    assert records[0]["step"] == 1
    assert "_ts" in records[0]


def test_json_logger_context_manager_and_overwrite(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    with U.JSONLogger(str(log_path)) as logger:
        logger.log(step=1)
    with U.JSONLogger(str(log_path)) as logger:
        logger.log(step=2)
    # default is append mode
    records = json.loads  # smoke: ensure json module usable
    all_records = U.JSONLogger(str(log_path)).read_all()
    assert len(all_records) == 2

    # now overwrite=True should truncate
    with U.JSONLogger(str(log_path), overwrite=True) as logger:
        logger.log(step=99)
    all_records2 = U.JSONLogger(str(log_path)).read_all()
    assert len(all_records2) == 1
    assert all_records2[0]["step"] == 99


def test_tensorboard_logger_noop_without_tensorboard(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if "tensorboard" in name:
            raise ImportError("forced for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with U.TensorBoardLogger(str(tmp_path)) as tb:
        assert tb._available is False
        tb.scalar("loss", 1.0, 0)   # should be a no-op, not raise
        tb.scalars("losses", {"a": 1.0}, 0)


def test_tensorboard_logger_with_real_backend_if_available(tmp_path):
    pytest.importorskip("torch.utils.tensorboard")
    tb = U.TensorBoardLogger(str(tmp_path))
    if tb._available:
        tb.scalar("loss", 0.1, 0)
        tb.scalars("group", {"a": 1.0, "b": 2.0}, 0)
        tb.close()


# ============================================================================
# 7. utils.py -- Dtype / device helpers
# ============================================================================

def test_to_float32_and_float64():
    x = np.array([1, 2, 3], dtype=np.int32)
    assert U.to_float32(x).dtype == np.float32
    assert U.to_float64(x).dtype == np.float64


def test_detect_devices_returns_expected_keys():
    devices = U.detect_devices()
    assert set(devices.keys()) == {"torch", "jax", "tensorflow"}
    for v in devices.values():
        assert isinstance(v, list)


@pytest.mark.parametrize(
    "dtype_str,expected",
    [
        ("float32", "float32"),
        ("<class 'numpy.float32'>", "float32"),
        ("float64", "float64"),
        ("float16", "float16"),
        ("int32", "int32"),
    ],
)
def test_get_dtype_string(dtype_str, expected):
    assert U.get_dtype_string("backend", dtype_str) == expected


# ============================================================================
# 8. utils.py -- Misc
# ============================================================================

def test_timer_context_manager_logs(caplog):
    with caplog.at_level(logging.INFO, logger="physai"):
        with U.timer("my-op"):
            time.sleep(0.001)
    assert any("my-op" in r.message for r in caplog.records)


def test_timer_default_label(caplog):
    with caplog.at_level(logging.INFO, logger="physai"):
        with U.timer():
            pass
    assert len(caplog.records) >= 1


def test_progress_bar_update(capsys):
    pb = U.ProgressBar(total=3, width=10, prefix="Test")
    pb.update(1, loss="1e-1")
    pb.update(2, loss="1e-2")
    pb.update(3, loss="1e-3")
    out = capsys.readouterr().out
    assert "Test" in out
    assert "3/3" in out


def test_progress_bar_zero_total_no_crash(capsys):
    pb = U.ProgressBar(total=0)
    pb.update(0)


def test_flatten_and_unflatten_dict_roundtrip():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = U.flatten_dict(nested)
    assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}
    back = U.unflatten_dict(flat)
    assert back == nested


def test_flatten_dict_custom_sep():
    nested = {"a": {"b": 1}}
    flat = U.flatten_dict(nested, sep="/")
    assert flat == {"a/b": 1}
    back = U.unflatten_dict(flat, sep="/")
    assert back == nested


def test_count_parameters_numpy():
    params = [np.zeros((3, 4)), np.zeros((5,)), np.zeros((2, 2, 2))]
    assert U.count_parameters_numpy(params) == 12 + 5 + 8


@pytest.mark.parametrize(
    "n,expected_suffix",
    [
        (500, ""),
        (1_500, "K"),
        (2_500_000, "M"),
        (3_500_000_000, "B"),
    ],
)
def test_format_number(n, expected_suffix):
    s = U.format_number(n)
    if expected_suffix:
        assert s.endswith(expected_suffix)
    else:
        assert s == str(int(n))


def test_pairwise_distance_self():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    D = U.pairwise_distance(X)
    assert D.shape == (2, 2)
    assert np.isclose(D[0, 1], 5.0)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_pairwise_distance_two_sets():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0], [0.0, 0.0]])
    D = U.pairwise_distance(X, Y)
    assert D.shape == (1, 2)
    assert np.isclose(D[0, 0], 5.0)
    assert np.isclose(D[0, 1], 0.0)


# ============================================================================
# 9. visualization.py
# ============================================================================

def test_plot_loss_history_basic():
    hist = _History()
    fig, ax = V.plot_loss_history(hist)
    assert fig is not None and ax is not None


def test_plot_loss_history_no_val_no_log_with_smoothing():
    hist = _History(with_val=False)
    fig, ax = V.plot_loss_history(hist, log_scale=False, smoothing=5, terms=["pde"])
    assert fig is not None


def test_plot_loss_history_missing_term_warns():
    hist = _History()
    with pytest.warns(UserWarning):
        V.plot_loss_history(hist, terms=["pde", "does_not_exist"])


def test_plot_solution_1d_with_reference_and_error():
    x = np.linspace(0, 1, 50)
    pred = np.sin(2 * np.pi * x)
    ref = pred + 0.01
    fig, axes = V.plot_solution_1d(x, pred, ref)
    assert fig is not None
    assert len(axes) == 2


def test_plot_solution_1d_no_reference():
    x = np.linspace(0, 1, 20)
    pred = np.cos(x)
    fig, ax = V.plot_solution_1d(x, pred)
    assert fig is not None


def test_plot_solution_1d_reference_but_no_error_panel():
    x = np.linspace(0, 1, 20)
    pred = np.cos(x)
    ref = pred + 0.1
    fig, ax = V.plot_solution_1d(x, pred, ref, show_error=False)
    assert fig is not None


def test_plot_solution_2d_default_and_symmetric():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 12)
    field = np.outer(np.sin(y * np.pi), np.cos(x * np.pi))
    fig, ax = V.plot_solution_2d(x, y, field)
    assert fig is not None
    fig2, ax2 = V.plot_solution_2d(x, y, field, symmetric_cbar=True, show_contours=False)
    assert fig2 is not None


def test_plot_solution_2d_comparison_linear_and_log_error():
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    pred = np.random.default_rng(6).normal(size=(8, 8))
    ref = pred + 0.05
    fig, axes = V.plot_solution_2d_comparison(x, y, pred, ref)
    assert len(axes) == 3
    fig2, axes2 = V.plot_solution_2d_comparison(x, y, pred, ref, error_log_scale=True)
    assert len(axes2) == 3


def test_plot_residual_field_log_and_linear():
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    residual = np.random.default_rng(7).normal(size=(8, 8))
    fig, ax = V.plot_residual_field(x, y, residual, log_scale=True)
    assert fig is not None
    fig2, ax2 = V.plot_residual_field(x, y, residual, log_scale=False)
    assert fig2 is not None


def test_plot_collocation_points_all_sets():
    rng = np.random.default_rng(8)
    collocation = rng.uniform(size=(100, 2))
    bc = rng.uniform(size=(20, 2))
    ic = rng.uniform(size=(15, 2))
    data = rng.uniform(size=(10, 2))
    fig, ax = V.plot_collocation_points(collocation, bc, ic, data)
    assert fig is not None


def test_plot_collocation_points_only_required():
    rng = np.random.default_rng(9)
    collocation = rng.uniform(size=(50, 2))
    fig, ax = V.plot_collocation_points(collocation)
    assert fig is not None


def test_plot_spectrum_1d():
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    field = np.sin(4 * x)
    fig, ax = V.plot_spectrum(field)
    assert fig is not None


def test_plot_spectrum_2d_with_reference_and_kolmogorov():
    rng = np.random.default_rng(10)
    field = rng.normal(size=(32, 32))
    ref = rng.normal(size=(32, 32))
    fig, ax = V.plot_spectrum(field, reference_field=ref, kolmogorov_slope=True)
    assert fig is not None


def test_animate_1d_solution_basic():
    x = np.linspace(0, 1, 30)
    frames = np.array([np.sin(x + t) for t in np.linspace(0, 1, 5)])
    anim = V.animate_1d_solution(x, frames)
    assert anim is not None


def test_animate_1d_solution_with_reference_and_params():
    x = np.linspace(0, 1, 20)
    frames = np.array([np.sin(x + t) for t in np.linspace(0, 1, 4)])
    ref_frames = frames + 0.05
    params = np.linspace(0, 1, 4)
    anim = V.animate_1d_solution(
        x, frames, reference_frames=ref_frames, param_values=params
    )
    assert anim is not None


def test_animate_1d_solution_save(tmp_path):
    pytest.importorskip("PIL")  # PillowWriter needs pillow
    x = np.linspace(0, 1, 10)
    frames = np.array([np.sin(x + t) for t in np.linspace(0, 1, 3)])
    out = tmp_path / "anim.gif"
    V.animate_1d_solution(x, frames, save_path=str(out), fps=5)
    assert out.exists()


def test_plot_training_animation_basic():
    x = np.linspace(0, 1, 20)
    snapshots = [np.sin(x + t) for t in np.linspace(0, 1, 4)]
    loss_history = [1.0, 0.5, 0.25, 0.1]
    anim = V.plot_training_animation(x, snapshots, loss_history)
    assert anim is not None


def test_plot_training_animation_with_custom_steps():
    x = np.linspace(0, 1, 15)
    snapshots = [np.cos(x + t) for t in np.linspace(0, 1, 3)]
    loss_history = [1.0, 0.3, 0.05]
    anim = V.plot_training_animation(
        x, snapshots, loss_history, snapshot_steps=[0, 10, 20]
    )
    assert anim is not None


def test_plot_error_convergence_basic():
    grid_sizes = [8, 16, 32, 64]
    errors = [1e-1, 3e-2, 8e-3, 2e-3]
    fig, ax = V.plot_error_convergence(grid_sizes, errors)
    assert fig is not None


def test_plot_error_convergence_with_reference_slopes():
    grid_sizes = [8, 16, 32, 64]
    errors = [1e-1, 3e-2, 8e-3, 2e-3]
    fig, ax = V.plot_error_convergence(
        grid_sizes, errors, reference_slopes={"order-2": -2.0, "order-1": -1.0}
    )
    assert fig is not None


def test_require_mpl_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(V, "_MPL_AVAILABLE", False)
    with pytest.raises(ImportError):
        V._require_mpl()


# ============================================================================
# 10. visualization_nd.py
# ============================================================================

def test_describe_field_basic():
    field = np.random.default_rng(11).normal(size=(4, 5, 6))
    desc = VN.describe_field(field)
    assert "shape=(4, 5, 6)" in desc
    assert "ndim=3" in desc


def test_describe_field_with_axis_names_and_coords():
    field = np.zeros((3, 4))
    coords = [np.linspace(0, 1, 3), np.linspace(-1, 1, 4)]
    desc = VN.describe_field(field, axis_names=["space", "time"])
    assert "space" in desc and "time" in desc


def test_make_axis_specs_errors():
    with pytest.raises(ValueError):
        VN._make_axis_specs(2, (3, 4), axis_names=["only_one"])
    with pytest.raises(ValueError):
        VN._make_axis_specs(2, (3, 4), coords=[np.arange(3)])
    with pytest.raises(ValueError):
        VN._make_axis_specs(1, (3,), coords=[np.arange(5)])


def test_slice_field_1d_and_2d():
    field = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    s1 = VN.slice_field(field, keep_axes=[0])
    assert s1.shape == (2,)
    s2 = VN.slice_field(field, keep_axes=[0, 2], fixed_indices={1: 1})
    assert s2.shape == (2, 4)


def test_slice_field_reorders_axes():
    field = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    s_normal = VN.slice_field(field, keep_axes=[0, 2])
    s_swapped = VN.slice_field(field, keep_axes=[2, 0])
    assert s_swapped.shape == (4, 2)
    assert np.array_equal(s_swapped, s_normal.T)


@pytest.mark.parametrize("reduction", ["mean", "max", "min", "sum", "rms", "std"])
def test_project_field_all_reductions(reduction):
    field = np.random.default_rng(12).normal(size=(3, 4, 5))
    proj = VN.project_field(field, keep_axes=[0], reduction=reduction)
    assert proj.shape == (3,)


def test_project_field_invalid_reduction():
    field = np.zeros((3, 3))
    with pytest.raises(ValueError):
        VN.project_field(field, keep_axes=[0], reduction="bogus")


def test_project_field_no_reduction_needed():
    field = np.zeros((3, 4))
    proj = VN.project_field(field, keep_axes=[0, 1])
    assert proj.shape == (3, 4)


def test_project_field_reorders_axes():
    field = np.random.default_rng(13).normal(size=(3, 4, 5))
    p_normal = VN.project_field(field, keep_axes=[0, 2])
    p_swapped = VN.project_field(field, keep_axes=[2, 0])
    assert np.allclose(p_swapped, p_normal.T)


def test_plot_slice_1d_slice_mode():
    field = np.random.default_rng(14).normal(size=(10, 6, 6))
    fig, ax = VN.plot_slice(field, keep_axes=[0], mode="slice")
    assert fig is not None


def test_plot_slice_2d_slice_mode_with_fixed_indices():
    field = np.random.default_rng(15).normal(size=(5, 6, 7))
    fig, ax = VN.plot_slice(field, keep_axes=[1, 2], fixed_indices={0: 2})
    assert fig is not None


def test_plot_slice_project_mode():
    field = np.random.default_rng(16).normal(size=(5, 6, 7))
    fig, ax = VN.plot_slice(field, keep_axes=[0, 1], mode="project", reduction="max")
    assert fig is not None


def test_plot_slice_1d_project_mode():
    field = np.random.default_rng(17).normal(size=(5, 6, 7))
    fig, ax = VN.plot_slice(field, keep_axes=[0], mode="project", reduction="rms")
    assert fig is not None


def test_plot_slice_invalid_keep_axes_length():
    field = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        VN.plot_slice(field, keep_axes=[0, 1, 2])


def test_plot_slice_invalid_mode():
    field = np.zeros((3, 3))
    with pytest.raises(ValueError):
        VN.plot_slice(field, keep_axes=[0], mode="bogus")


def test_plot_slice_asymmetric_cbar_and_title():
    field = np.random.default_rng(18).normal(size=(4, 4))
    fig, ax = VN.plot_slice(
        np.expand_dims(field, 0),
        keep_axes=[1, 2],
        symmetric_cbar=False,
        title="Custom title",
    )
    assert fig is not None


def test_plot_slice_grid_1d_and_2d():
    field = np.random.default_rng(19).normal(size=(4, 8, 8))
    fig, axes = VN.plot_slice_grid(field, keep_axes=[1], sweep_axis=0, n_panels=4)
    assert fig is not None
    fig2, axes2 = VN.plot_slice_grid(field, keep_axes=[1, 2], sweep_axis=0, n_panels=4)
    assert fig2 is not None


def test_plot_slice_grid_invalid_sweep_axis():
    field = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        VN.plot_slice_grid(field, keep_axes=[0], sweep_axis=0)


def test_plot_isosurface_3d_matplotlib_fallback():
    field = np.random.default_rng(20).normal(size=(10, 10, 10))
    import physai.visualization_nd as vn_mod
    original = vn_mod._PLOTLY_AVAILABLE
    vn_mod._PLOTLY_AVAILABLE = False
    try:
        with pytest.warns(UserWarning):
            fig, ax = VN.plot_isosurface_3d(field)
        assert fig is not None
    finally:
        vn_mod._PLOTLY_AVAILABLE = original


def test_plot_isosurface_3d_with_plotly_if_available():
    pytest.importorskip("plotly")
    field = np.random.default_rng(21).normal(size=(8, 8, 8))
    fig = VN.plot_isosurface_3d(field, mode="isosurface")
    assert fig is not None
    fig2 = VN.plot_isosurface_3d(field, mode="volume")
    assert fig2 is not None


def test_plot_isosurface_3d_invalid_mode_with_plotly():
    pytest.importorskip("plotly")
    field = np.random.default_rng(22).normal(size=(6, 6, 6))
    with pytest.raises(ValueError):
        VN.plot_isosurface_3d(field, mode="bogus")


def test_plot_isosurface_3d_wrong_ndim():
    field = np.zeros((3, 3))
    with pytest.raises(ValueError):
        VN.plot_isosurface_3d(field)


def test_animate_nd_field_1d_slice_mode():
    field = np.random.default_rng(23).normal(size=(4, 10))
    anim = VN.animate_nd_field(field, keep_axes=[1], sweep_axis=0)
    assert anim is not None


def test_animate_nd_field_2d_slice_mode():
    field = np.random.default_rng(24).normal(size=(3, 6, 6))
    anim = VN.animate_nd_field(field, keep_axes=[1, 2], sweep_axis=0)
    assert anim is not None


def test_animate_nd_field_project_mode():
    field = np.random.default_rng(25).normal(size=(3, 4, 6, 6))
    anim = VN.animate_nd_field(
        field, keep_axes=[2, 3], sweep_axis=0, mode="project", reduction="mean"
    )
    assert anim is not None


def test_animate_nd_field_invalid_args():
    field = np.zeros((3, 4, 4))
    with pytest.raises(ValueError):
        VN.animate_nd_field(field, keep_axes=[0], sweep_axis=0)
    with pytest.raises(ValueError):
        VN.animate_nd_field(field, keep_axes=[0, 1, 2], sweep_axis=None or 1)
    with pytest.raises(ValueError):
        VN.animate_nd_field(field, keep_axes=[0], sweep_axis=1, mode="bogus")


def test_animate_nd_field_save(tmp_path):
    pytest.importorskip("PIL")
    field = np.random.default_rng(26).normal(size=(3, 6))
    out = tmp_path / "nd_anim.gif"
    VN.animate_nd_field(field, keep_axes=[1], sweep_axis=0, save_path=str(out), fps=5)
    assert out.exists()


def test_animate_isosurface_3d_requires_plotly_or_runs():
    field_4d = np.random.default_rng(27).normal(size=(4, 4, 4, 3))
    try:
        import plotly  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            VN.animate_isosurface_3d(field_4d)
        return
    fig = VN.animate_isosurface_3d(field_4d)
    assert fig is not None


def test_animate_isosurface_3d_wrong_ndim():
    pytest.importorskip("plotly")
    field = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        VN.animate_isosurface_3d(field)


def test_interactive_nd_explorer_falls_back_without_ipywidgets(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "ipywidgets":
            raise ImportError("forced for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    field = np.random.default_rng(28).normal(size=(5, 6, 6))
    with pytest.warns(UserWarning):
        result = VN.interactive_nd_explorer(field, keep_axes=(1, 2))
    assert result is not None


def test_require_plotly_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(VN, "_PLOTLY_AVAILABLE", False)
    with pytest.raises(ImportError):
        VN._require_plotly()


def test_require_mpl_raises_when_unavailable_nd(monkeypatch):
    monkeypatch.setattr(VN, "_MPL_AVAILABLE", False)
    with pytest.raises(ImportError):
        VN._require_mpl()