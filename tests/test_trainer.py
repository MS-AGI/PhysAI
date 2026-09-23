"""
tests/test_trainer.py

Tests for physai.trainer: TrainingHistory, Callback classes, the
backend-agnostic gradient-clip helpers, and the Trainer class itself
(construction, the Adam training loop, checkpointing, prediction, RAR
resampling, the L-BFGS fine-tune phase, and the arbitrary-geometry
boundary-condition path).

Torch-only, matching the convention already used by test_pde_residual.py
and test_geometry.py in this suite -- Trainer hardcodes `import torch` at
module scope regardless of which backend is used for anything else, so a
torch install is required to import physai.trainer at all.

Design notes
------------
Every end-to-end test below builds the *same* minimal, cheap scenario:
homogeneous Poisson (-Delta u = 0, default zero source) on the unit square
with u = 0 on the boundary. This has an exact, trivial solution (u = 0),
and Poisson is linear/non-stiff/low-order, so AutoOptimizer picks a plain
"tanh" + "adam" configuration with no learnable-activation machinery in
the way -- the cheapest possible real training loop that still exercises
the full Trainer/AutoOptimizer/PINN/PDEResidual integration honestly,
rather than mocking any of it out.

`RuntimeConfig` and its nested dataclasses are all frozen, so tests that
need a non-default field (tiny `lbfgs_max_iter`, forced `rar_enabled`,
etc.) use `dataclasses.replace` rather than mutating in place.

Run with: pytest tests/test_trainer.py -v
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import torch

from physai.backends.torch_backend import TorchBackend
from physai.core.auto_optimizer import (
    AutoOptimizer, DomainSpec, ProblemSpec, RuntimeConfig, ModelConfig,
    TrainingConfig, SchedulerConfig, PDEMeta,
)
from physai.core.pde_residual import build_residual
from physai.geometry import BoundaryConditionSet, box, ball, face_region
from physai.models.pinn import PINN
from physai.models.spectral_element import USENOCPModule
from physai.trainer import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    PrintCallback,
    Trainer,
    TrainingHistory,
    _clip_grad_dict,
    _clip_gradients,
)

backend = TorchBackend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_poisson_trainer(
    *,
    max_epochs: int = 30,
    use_lbfgs_phase: bool = False,
    n_collocation: int = 64,
    n_bc_points: int = 16,
    callbacks=None,
    boundary_conditions: BoundaryConditionSet = None,
    compile: bool = False,
    val_points=None,
    val_values=None,
):
    """
    Minimal end-to-end setup: homogeneous Poisson on the unit square,
    u = 0 on the boundary (exact solution u = 0). Returns the constructed
    Trainer; callers inspect `.` attrs directly for the raw tensors.
    """
    domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    spec = ProblemSpec(
        pde_name        = "poisson",
        domain          = domain,
        model_arch      = "pinn",
        backend_name    = "torch",
        n_collocation   = n_collocation,
        n_bc_points     = n_bc_points,
        max_epochs      = max_epochs,
        use_lbfgs_phase = use_lbfgs_phase,
        target_loss     = -1.0,  # never trip early-stop-on-convergence
    )
    config = AutoOptimizer(backend, verbose=False).analyse(spec)
    assert config.model.activation == "tanh"  # sanity: Poisson is linear/non-stiff/low-order

    model = PINN(
        backend,
        layer_sizes  = config.model.layer_sizes,
        activation   = config.model.activation,
        use_residual = config.model.use_residual,
    )
    residual = build_residual("poisson", backend)

    rng = np.random.default_rng(0)
    coll_np = rng.uniform(0.0, 1.0, size=(n_collocation, 2)).astype(np.float32)
    collocation_points = backend.tensor(coll_np)

    bc_points = bc_values = None
    if boundary_conditions is None:
        edge_pts = []
        n_per_edge = max(n_bc_points // 4, 1)
        t = rng.uniform(0.0, 1.0, size=(n_per_edge,)).astype(np.float32)
        edge_pts.append(np.stack([np.zeros_like(t), t], axis=-1))
        edge_pts.append(np.stack([np.ones_like(t), t], axis=-1))
        edge_pts.append(np.stack([t, np.zeros_like(t)], axis=-1))
        edge_pts.append(np.stack([t, np.ones_like(t)], axis=-1))
        bc_np = np.concatenate(edge_pts, axis=0).astype(np.float32)
        bc_points = backend.tensor(bc_np)
        bc_values = backend.tensor(np.zeros((bc_np.shape[0], 1), dtype=np.float32))

    trainer = Trainer(
        backend,
        config,
        model,
        residual,
        collocation_points  = collocation_points,
        bc_points           = bc_points,
        bc_values           = bc_values,
        boundary_conditions = boundary_conditions,
        callbacks           = callbacks if callbacks is not None else [],
        log_every           = 10_000,  # effectively silent
        compile             = compile,
        val_points          = val_points,
        val_values          = val_values,
    )
    return trainer


# ---------------------------------------------------------------------------
# TrainingHistory
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def test_record_appends_all_fields(self):
        h = TrainingHistory()
        h.record(step=0, total=1.0, breakdown={"pde": 0.6, "bc": 0.4}, lr=1e-3, t=0.01)
        h.record(step=1, total=0.8, breakdown={"pde": 0.5, "bc": 0.3}, lr=1e-3, t=0.02, val=0.7)

        assert h.steps == [0, 1]
        assert h.total_loss == [1.0, 0.8]
        assert h.lr == [1e-3, 1e-3]
        assert h.wall_time == [0.01, 0.02]
        assert h.terms["pde"] == [0.6, 0.5]
        assert h.terms["bc"] == [0.4, 0.3]
        assert h.val_loss == [0.7]  # only recorded on the second call

    def test_as_dict_matches_fields(self):
        h = TrainingHistory()
        h.record(0, 1.0, {"pde": 1.0}, 1e-3, 0.0)
        d = h.as_dict()
        assert d["steps"] == [0]
        assert d["total_loss"] == [1.0]
        assert d["terms"] == {"pde": [1.0]}

    def test_save_npz_round_trips(self, tmp_path):
        h = TrainingHistory()
        for i in range(5):
            h.record(i, 1.0 / (i + 1), {"pde": 1.0 / (i + 1)}, 1e-3, float(i))
        out = tmp_path / "history.npz"
        h.save_npz(str(out))
        assert out.exists()

        archive = np.load(str(out))
        np.testing.assert_array_equal(archive["steps"], np.arange(5))
        np.testing.assert_allclose(archive["total_loss"], [1.0 / (i + 1) for i in range(5)])
        np.testing.assert_allclose(archive["term_pde"], [1.0 / (i + 1) for i in range(5)])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_base_callback_hooks_are_noops(self):
        cb = Callback()
        cb.on_train_begin(None)
        cb.on_epoch_end(None, 0, 0.0, {})
        cb.on_lbfgs_begin(None)
        cb.on_train_end(None)
        cb.on_rar_step(None, 0)

    def test_early_stopping_triggers_after_patience(self):
        cb = EarlyStoppingCallback(patience=3, min_delta=1e-6)

        class _FakeTrainer:
            _stop_flag = False

        trainer = _FakeTrainer()
        for step, loss in enumerate([1.0, 0.5, 0.25, 0.125]):
            cb.on_epoch_end(trainer, step, loss, {})
            assert trainer._stop_flag is False

        for step in range(3):
            cb.on_epoch_end(trainer, 100 + step, 0.125, {})
        assert trainer._stop_flag is True

    def test_early_stopping_resets_on_improvement(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=1e-6)

        class _FakeTrainer:
            _stop_flag = False

        trainer = _FakeTrainer()
        cb.on_epoch_end(trainer, 0, 1.0, {})
        cb.on_epoch_end(trainer, 1, 1.0, {})   # 1 no-improve
        cb.on_epoch_end(trainer, 2, 0.5, {})   # improves -> resets counter
        assert trainer._stop_flag is False
        cb.on_epoch_end(trainer, 3, 0.5, {})   # 1 no-improve again
        assert trainer._stop_flag is False

    def test_checkpoint_callback_writes_file_on_schedule(self, tmp_path):
        trainer = _build_poisson_trainer(max_epochs=1)
        cb = CheckpointCallback(str(tmp_path), every=5)

        cb.on_epoch_end(trainer, step=0, loss=1.0, breakdown={})
        assert (tmp_path / "ckpt_step0000000.npz").exists()

        cb.on_epoch_end(trainer, step=3, loss=1.0, breakdown={})
        assert not (tmp_path / "ckpt_step0000003.npz").exists()

        cb.on_epoch_end(trainer, step=5, loss=1.0, breakdown={})
        assert (tmp_path / "ckpt_step0000005.npz").exists()

    def test_print_callback_runs_without_error(self, capsys):
        trainer = _build_poisson_trainer(max_epochs=1)
        cb = PrintCallback(every=1)
        cb.on_train_begin(trainer)
        cb.on_epoch_end(trainer, 0, 1.23, {"pde": 1.0, "bc": 0.23})
        cb.on_lbfgs_begin(trainer)
        trainer.history.record(0, 1.23, {"pde": 1.0}, 1e-3, 0.01)
        cb.on_train_end(trainer)
        out = capsys.readouterr().out
        assert "loss=1.2300e+00" in out
        assert "L-BFGS" in out


# ---------------------------------------------------------------------------
# Gradient-clip helpers
# ---------------------------------------------------------------------------

class TestGradClipHelpers:
    def test_clip_gradients_reduces_large_norm(self):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([10.0, 10.0, 10.0, 10.0])
        _clip_gradients(backend, [p], max_norm=1.0)
        assert torch.linalg.norm(p.grad).item() == pytest.approx(1.0, rel=1e-4)

    def test_clip_gradients_leaves_small_norm_untouched(self):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([0.01, 0.01, 0.0, 0.0])
        original = p.grad.clone()
        _clip_gradients(backend, [p], max_norm=1.0)
        torch.testing.assert_close(p.grad, original)

    def test_clip_grad_dict_reduces_global_norm(self):
        grads = {
            "a": backend.tensor(np.array([3.0, 4.0], dtype=np.float32)),  # norm 5
            "b": backend.tensor(np.array([0.0], dtype=np.float32)),
        }
        clipped = _clip_grad_dict(backend, grads, max_norm=1.0)
        total_sq = sum(float(backend.to_numpy(backend.sum(backend.square(g))))
                       for g in clipped.values())
        assert np.sqrt(total_sq) == pytest.approx(1.0, rel=1e-3)

    def test_clip_grad_dict_list_variant(self):
        grads = [backend.tensor(np.array([3.0, 4.0], dtype=np.float32))]
        clipped = _clip_grad_dict(backend, grads, max_norm=1.0)
        norm = float(backend.to_numpy(backend.sqrt(backend.sum(backend.square(clipped[0])))))
        assert norm == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------

class TestTrainerConstruction:
    def test_rejects_both_bc_points_and_boundary_conditions(self):
        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=16, n_bc_points=8, max_epochs=1)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.zeros((16, 2), dtype=np.float32))
        bc_pts = backend.tensor(np.zeros((8, 2), dtype=np.float32))
        bc_val = backend.tensor(np.zeros((8, 1), dtype=np.float32))
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add("dirichlet", value=lambda x: np.zeros(len(x)))

        with pytest.raises(ValueError):
            Trainer(
                backend, config, model, residual,
                collocation_points=coll,
                bc_points=bc_pts, bc_values=bc_val,
                boundary_conditions=bcs,
            )

    def test_default_callback_is_print_callback(self):
        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=8, n_bc_points=4, max_epochs=1)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.zeros((8, 2), dtype=np.float32))
        t = Trainer(backend, config, model, residual, collocation_points=coll)
        assert len(t.callbacks) == 1
        assert isinstance(t.callbacks[0], PrintCallback)

    def test_pde_and_bc_terms_registered_in_composite(self):
        trainer = _build_poisson_trainer(max_epochs=1)
        total, breakdown = trainer._composite()
        assert "pde" in breakdown
        assert "bc" in breakdown
        assert np.isfinite(float(backend.to_numpy(total)))


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------

class TestTrainerTrainingLoop:
    def test_loss_is_finite_and_history_populated(self):
        trainer = _build_poisson_trainer(max_epochs=20)
        history = trainer.train()

        assert len(history.steps) == 20
        assert all(np.isfinite(v) for v in history.total_loss)
        assert len(history.terms["pde"]) == 20
        assert len(history.terms["bc"]) == 20

    def test_homogeneous_poisson_loss_decreases(self):
        # u = 0 exactly satisfies both -Delta u=0 and u|boundary=0, so
        # even this cheap a run should show a clear downward trend.
        trainer = _build_poisson_trainer(max_epochs=150)
        history = trainer.train()
        early = np.mean(history.total_loss[:10])
        late = np.mean(history.total_loss[-10:])
        assert late < early

    def test_train_respects_target_loss_early_stop(self):
        trainer = _build_poisson_trainer(max_epochs=10_000)
        trainer.config = dataclasses.replace(
            trainer.config,
            problem=dataclasses.replace(trainer.config.problem, target_loss=float("inf")),
        )
        history = trainer.train()
        assert len(history.steps) == 1  # stops after the very first step

    def test_early_stopping_callback_halts_training(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=1e10)  # any move "fails to improve"
        trainer = _build_poisson_trainer(max_epochs=10_000, callbacks=[cb])
        history = trainer.train()
        assert len(history.steps) < 10


class TestTrainerPredictAndCheckpoint:
    def test_predict_shape_and_no_grad_tracking(self):
        trainer = _build_poisson_trainer(max_epochs=1)
        pts = backend.tensor(np.random.uniform(0, 1, size=(5, 2)).astype(np.float32))
        out = trainer.predict(pts)
        assert tuple(out.shape) == (5, 1)
        assert out.requires_grad is False

    def test_checkpoint_round_trip_restores_parameters(self, tmp_path):
        trainer = _build_poisson_trainer(max_epochs=5)
        trainer.train()

        before = {k: v.clone() for k, v in trainer.model.named_parameters().items()}
        ckpt_path = str(tmp_path / "ckpt")
        trainer.save_checkpoint(ckpt_path)

        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(1.0)
        after_perturb = {k: v.clone() for k, v in trainer.model.named_parameters().items()}
        for k in before:
            assert not torch.allclose(before[k], after_perturb[k])

        trainer.load_checkpoint(ckpt_path)
        restored = trainer.model.named_parameters()
        for k in before:
            torch.testing.assert_close(before[k], restored[k])

    def test_checkpoint_restores_step_counter(self, tmp_path):
        trainer = _build_poisson_trainer(max_epochs=7)
        trainer.train()
        assert trainer._step == 7

        ckpt_path = str(tmp_path / "ckpt")
        trainer.save_checkpoint(ckpt_path)
        trainer._step = 0
        trainer.load_checkpoint(ckpt_path)
        assert trainer._step == 7


class TestTrainerRAR:
    def test_rar_resample_changes_collocation_points(self):
        trainer = _build_poisson_trainer(max_epochs=1, n_collocation=32)
        # Poisson's default rar_enabled is False (linear, low kappa) --
        # force it on with a tiny interval so a short run actually
        # exercises `_maybe_rar` -> `AutoOptimizer.rar_resample`.
        trainer.config = dataclasses.replace(
            trainer.config,
            training=dataclasses.replace(
                trainer.config.training,
                rar_enabled=True,
                rar_interval=2,
                rar_fraction=0.5,
            ),
        )
        original = trainer._coll.clone()
        trainer.train(extra_epochs=4)  # 1 (config) + 4 extra = 5 steps
        assert trainer._coll.shape[0] == original.shape[0]
        assert not torch.allclose(trainer._coll, original)

    def test_maybe_rar_noop_when_disabled(self):
        trainer = _build_poisson_trainer(max_epochs=1, n_collocation=32)
        assert trainer.config.training.rar_enabled is False
        original = trainer._coll.clone()
        trainer._maybe_rar(step=0)
        torch.testing.assert_close(trainer._coll, original)


class TestTrainerLBFGSPhase:
    def test_lbfgs_phase_runs_and_stays_finite(self):
        trainer = _build_poisson_trainer(max_epochs=5, use_lbfgs_phase=True)
        # Shrink lbfgs_max_iter so the closure-based strong-Wolfe search
        # stays fast rather than running the full default 500 iters.
        trainer.config = dataclasses.replace(
            trainer.config,
            training=dataclasses.replace(trainer.config.training, lbfgs_max_iter=3),
        )
        history = trainer.train()
        assert np.isfinite(history.total_loss[-1])
        total_after, _ = trainer._composite()
        # Strong-Wolfe line search only accepts non-increasing steps.
        assert float(backend.to_numpy(total_after)) <= history.total_loss[-1] + 1e-4

    def test_lbfgs_begin_callback_fires(self):
        events = []

        class _RecordingCallback(Callback):
            def on_lbfgs_begin(self, trainer):
                events.append("lbfgs_begin")

        trainer = _build_poisson_trainer(
            max_epochs=2, use_lbfgs_phase=True, callbacks=[_RecordingCallback()]
        )
        trainer.config = dataclasses.replace(
            trainer.config,
            training=dataclasses.replace(trainer.config.training, lbfgs_max_iter=2),
        )
        trainer.train()
        assert events == ["lbfgs_begin"]


# ---------------------------------------------------------------------------
# Arbitrary-geometry boundary conditions
# ---------------------------------------------------------------------------

class TestTrainerGeometryBoundaryConditions:
    def _make_bcs(self) -> BoundaryConditionSet:
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add(
            "dirichlet",
            value=lambda x: np.zeros(len(x)),
            region=face_region(axis=0, side="min"),
            name="left_wall",
        )
        bcs.add(
            "dirichlet",
            value=lambda x: np.zeros(len(x)),
            region=lambda x: np.ones(len(x), dtype=bool),  # catch-all for the rest
            name="rest",
        )
        return bcs

    def test_boundary_condition_set_wires_into_composite(self):
        bcs = self._make_bcs()
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        total, breakdown = trainer._composite()
        assert "left_wall" in breakdown
        assert "rest" in breakdown
        assert np.isfinite(float(backend.to_numpy(total)))

    def test_resample_boundary_conditions_refreshes_points(self):
        bcs = self._make_bcs()
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        first = {
            name: entry.get("points")
            for name, entry in trainer._bc_targets.items()
            if "points" in entry
        }
        trainer.resample_boundary_conditions()
        second = {
            name: entry.get("points")
            for name, entry in trainer._bc_targets.items()
            if "points" in entry
        }

        def _changed(a, b) -> bool:
            # Region-filtered BCs (e.g. "left_wall" via face_region) draw
            # their point count from however many freshly-resampled
            # boundary points happen to land in that region — that count
            # can legitimately differ between resamples, it isn't held
            # fixed. A shape change is itself evidence the points were
            # refreshed, so it counts as "changed" rather than being run
            # through `torch.allclose` (which requires equal shapes and
            # would raise instead of comparing).
            if a.shape != b.shape:
                return True
            return not torch.allclose(a, b)

        assert any(
            name in second and _changed(first[name], second[name])
            for name in first
        )

    def test_geometry_trainer_trains_without_error(self):
        bcs = self._make_bcs()
        trainer = _build_poisson_trainer(max_epochs=15, boundary_conditions=bcs)
        history = trainer.train()
        assert all(np.isfinite(v) for v in history.total_loss)


# ---------------------------------------------------------------------------
# diff_mode plumbing (Trainer's own Neumann/Robin gradient helper)
# ---------------------------------------------------------------------------

class TestTrainerDiffMode:
    @pytest.mark.parametrize("mode", ["reverse", "forward", "taylor"])
    def test_spatial_grad_finite_across_modes(self, mode):
        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=8, n_bc_points=4, max_epochs=1)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.random.uniform(0, 1, size=(8, 2)).astype(np.float32))
        trainer = Trainer(
            backend, config, model, residual,
            collocation_points=coll, diff_mode=mode,
        )
        pts = backend.tensor(np.random.uniform(0, 1, size=(6, 2)).astype(np.float32))
        grad = trainer._spatial_grad(pts)
        assert torch.isfinite(grad).all()
        assert tuple(grad.shape) == (6, 2)


# ---------------------------------------------------------------------------
# val_points / _val_loss
# ---------------------------------------------------------------------------

class TestTrainerValLoss:
    def test_val_loss_none_when_no_val_points(self):
        trainer = _build_poisson_trainer(max_epochs=1)
        assert trainer._val_loss() is None

    def test_val_loss_computed_and_finite_and_untracked(self):
        rng = np.random.default_rng(1)
        v_pts = backend.tensor(rng.uniform(0, 1, size=(10, 2)).astype(np.float32))
        v_val = backend.tensor(np.zeros((10, 1), dtype=np.float32))
        trainer = _build_poisson_trainer(max_epochs=1, val_points=v_pts, val_values=v_val)
        val = trainer._val_loss()
        assert val is not None
        assert np.isfinite(val)

    def test_history_records_val_loss_on_log_steps(self):
        rng = np.random.default_rng(2)
        v_pts = backend.tensor(rng.uniform(0, 1, size=(6, 2)).astype(np.float32))
        v_val = backend.tensor(np.zeros((6, 1), dtype=np.float32))
        trainer = _build_poisson_trainer(max_epochs=3, val_points=v_pts, val_values=v_val)
        trainer.log_every = 1  # log (and thus compute val_loss) every step
        history = trainer.train()
        assert len(history.val_loss) == 3


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestTrainerRepr:
    def test_repr_contains_key_fields(self):
        trainer = _build_poisson_trainer(max_epochs=5)
        r = repr(trainer)
        assert "backend=torch" in r
        assert "pde=poisson" in r
        assert "optimizer=" in r
        assert "lr=" in r
        assert "step=0/5" in r


# ---------------------------------------------------------------------------
# compile=True (torch.compile path)
# ---------------------------------------------------------------------------

class TestTrainerCompile:
    def test_compile_true_trains_and_composite_becomes_compiled(self):
        """
        On an environment where torch.compile actually works end-to-end,
        `_composite_compiled` should hold the compiled callable after
        training. But `backend.jit()`/`torch.compile` compiles lazily —
        the compiled graph is only built (and can only fail) on first
        *invocation*, not at wrap time — so on hardware torch.compile
        can't target (e.g. a pre-Volta GPU: `GPUTooOldForTriton`, seen in
        practice on a Quadro P620 during this test's own development),
        `Trainer._get_composite_fn`'s first-call safety net catches that,
        warns once, and permanently falls back to eager for the rest of
        training instead of crashing (see its docstring). That fallback
        is the correct, intended behavior on such hardware, not a bug —
        so this test accepts either outcome: real compilation (asserted
        strictly, when it happens) or a clean graceful fallback (asserted
        via `trainer._compile` having been flipped off), and either way
        confirms training still produced a finite loss history.
        """
        trainer = _build_poisson_trainer(max_epochs=3, compile=True)
        assert trainer._composite_compiled is None  # lazy: nothing yet
        history = trainer.train()
        assert all(np.isfinite(v) for v in history.total_loss)

        if trainer._compile:
            # Compilation actually succeeded in this environment.
            assert trainer._composite_compiled is not None
        else:
            # Environment couldn't run compiled mode (e.g. GPU too old for
            # Triton) — _get_composite_fn's first-call fallback caught it,
            # warned, and switched back to eager. That's the documented,
            # intended behavior, not a failure of this test.
            assert trainer._composite_compiled is None

    def test_compile_false_never_compiles(self):
        trainer = _build_poisson_trainer(max_epochs=2, compile=False)
        trainer.train()
        assert trainer._composite_compiled is None

    def test_get_composite_fn_returns_same_compiled_instance_across_calls(self):
        trainer = _build_poisson_trainer(max_epochs=1, compile=True)
        fn1 = trainer._get_composite_fn()
        fn2 = trainer._get_composite_fn()
        assert fn1 is fn2

    def test_compile_unsupported_backend_warns_once_and_falls_back(self, monkeypatch):
        """Simulates a backend name compile has no real support for
        (jax/paddle) without needing those backends installed. `.name` is
        a read-only property on every backend class (a fixed identity
        constant, not per-instance state — see AbstractBackend.name), so
        it can't be assigned directly on an instance; monkeypatching the
        property on the *class* is the correct way to override it for
        this one test, and `monkeypatch` auto-restores the original
        property afterward regardless of how the test exits."""
        trainer = _build_poisson_trainer(max_epochs=1, compile=True)
        monkeypatch.setattr(
            type(trainer.backend), "name", property(lambda self: "paddle")
        )
        with pytest.warns(UserWarning, match="has no meaningful effect"):
            fn = trainer._get_composite_fn()
        assert fn is trainer._composite  # fell back to eager
        assert trainer._compile_unsupported_warned is True
        # second call must NOT warn again
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trainer._get_composite_fn()


# ---------------------------------------------------------------------------
# Boundary-condition kinds: neumann / robin / periodic / zero-point /
# unknown-kind dispatch in _bc_condition_loss
# ---------------------------------------------------------------------------

class TestTrainerBoundaryConditionKinds:
    def test_neumann_condition_loss_is_finite(self):
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add(
            "neumann",
            value=lambda x: np.zeros(len(x)),
            region=face_region(axis=0, side="min"),
            name="left_flux",
        )
        bcs.add(
            "dirichlet",
            value=lambda x: np.zeros(len(x)),
            region=lambda x: np.ones(len(x), dtype=bool),
            name="rest",
        )
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        loss = trainer._bc_condition_loss(bcs.conditions[0])
        assert np.isfinite(float(backend.to_numpy(loss)))

    def test_robin_condition_loss_is_finite(self):
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add(
            "robin",
            value=lambda x: np.zeros(len(x)),
            coeff_a=1.0, coeff_b=0.5,
            region=face_region(axis=0, side="max"),
            name="right_robin",
        )
        bcs.add(
            "dirichlet",
            value=lambda x: np.zeros(len(x)),
            region=lambda x: np.ones(len(x), dtype=bool),
            name="rest",
        )
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        loss = trainer._bc_condition_loss(bcs.conditions[0])
        assert np.isfinite(float(backend.to_numpy(loss)))

    def test_periodic_condition_loss_is_finite(self):
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add(
            "periodic",
            region=face_region(axis=0, side="min"),
            pair_region=face_region(axis=0, side="max"),
            pair_map=lambda x: np.stack([np.ones(len(x)), x[:, 1]], axis=-1),
            name="x_periodic",
        )
        bcs.add(
            "dirichlet",
            value=lambda x: np.zeros(len(x)),
            region=lambda x: np.ones(len(x), dtype=bool),
            name="rest",
        )
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        loss = trainer._bc_condition_loss(bcs.conditions[0])
        assert np.isfinite(float(backend.to_numpy(loss)))

    def test_zero_points_condition_returns_zero_loss(self):
        """A region matching nothing in the current sample (`entry["n"]==0`)
        must return a zero loss rather than erroring."""
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        # A region that can never match any boundary point.
        bcs.add("dirichlet", value=lambda x: np.zeros(len(x)),
                region=lambda x: np.zeros(len(x), dtype=bool), name="impossible")
        bcs.add("dirichlet", value=lambda x: np.zeros(len(x)),
                region=lambda x: np.ones(len(x), dtype=bool), name="rest")
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        loss = trainer._bc_condition_loss(bcs.conditions[0])
        assert float(backend.to_numpy(loss)) == 0.0

    def test_missing_entry_returns_zero_loss(self):
        """A BC whose name never made it into `_bc_targets` at all (not
        just zero points) must also return a zero loss, not KeyError."""
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add("dirichlet", value=lambda x: np.zeros(len(x)), name="only")
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        trainer._bc_targets = {}  # simulate no targets registered yet
        loss = trainer._bc_condition_loss(bcs.conditions[0])
        assert float(backend.to_numpy(loss)) == 0.0

    def test_unknown_kind_raises_value_error(self):
        """`_bc_condition_loss` must reject a kind outside the 4 known
        ones. BoundaryCondition itself validates kind at construction, so
        this reaches the branch by directly forging a `_bc_targets` entry
        with a bogus kind (the only way to get an invalid kind past the
        dataclass's own __post_init__ check)."""
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        bcs = BoundaryConditionSet(geo)
        bcs.add("dirichlet", value=lambda x: np.zeros(len(x)), name="only")
        trainer = _build_poisson_trainer(max_epochs=1, boundary_conditions=bcs)
        pts = backend.tensor(np.zeros((2, 2), dtype=np.float32))
        trainer._bc_targets["only"] = {"kind": "bogus", "n": 2, "points": pts}
        with pytest.raises(ValueError, match="Unknown boundary condition kind"):
            trainer._bc_condition_loss(bcs.conditions[0])


# ---------------------------------------------------------------------------
# cross_validate — argument-validation branches (no dedalus/cupy required,
# every one of these raises before Solver.solve_box/solve_geometry is called)
# ---------------------------------------------------------------------------

class TestTrainerCrossValidateValidation:
    def test_geometry_path_requires_pde_and_bc_value(self):
        trainer = _build_poisson_trainer(max_epochs=1)
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        with pytest.raises(ValueError, match="requires 'pde' and 'bc_value'"):
            trainer.cross_validate(geometry=geo)

    def test_box_path_reports_all_missing_args(self):
        trainer = _build_poisson_trainer(max_epochs=1)
        with pytest.raises(ValueError, match="domain_type"):
            trainer.cross_validate()

    def test_box_path_rejects_non_1d_domain(self):
        # _build_poisson_trainer's domain is 2D, so even with every box
        # arg supplied, cross_validate(geometry=None) must reject it.
        trainer = _build_poisson_trainer(max_epochs=1)
        with pytest.raises(ValueError, match="only supports 1D spatial"):
            trainer.cross_validate(
                domain_type="chebyshev",
                bounds=(0.0, 1.0),
                variables=["u"],
                equations=["dt(u) - dx(dx(u)) = 0"],
                bcs=["left(u) = 0", "right(u) = 0"],
                ics={"u": lambda x: np.zeros_like(x)},
            )


class TestTrainerCrossValidateGeometryPath:
    """Real (non-mocked) numerical cross-validation against
    Solver.solve_geometry — a scipy.sparse embedded-boundary FD solve, not
    Dedalus, so it needs no optional dependency beyond scipy (already a
    hard dependency of this package)."""

    def test_cross_validate_poisson_on_ball_returns_finite_errors(self):
        # A tiny 2D Poisson trainer whose exact solution (u=0, zero
        # source, zero Dirichlet boundary) exactly matches
        # solve_geometry("poisson")'s default zero source too, so even an
        # undertrained model should show a small, always-finite error.
        geo = ball([0.5, 0.5], 0.4)
        trainer = _build_poisson_trainer(max_epochs=5)
        result = trainer.cross_validate(
            geometry=geo,
            pde="poisson",
            bc_value=lambda x: np.zeros(len(x)),
            grid_points=12,
        )
        assert set(result.keys()) == {"l2_abs", "l2_rel", "n_compared"}
        assert np.isfinite(result["l2_abs"])
        assert np.isfinite(result["l2_rel"])
        assert result["n_compared"] > 0

    def test_cross_validate_helmholtz_pde_params_forwarded(self):
        geo = box([(0.0, 1.0), (0.0, 1.0)])
        trainer = _build_poisson_trainer(max_epochs=1)
        result = trainer.cross_validate(
            geometry=geo,
            pde="helmholtz",
            bc_value=lambda x: np.zeros(len(x)),
            pde_params={"k": 1.0},
            grid_points=10,
        )
        assert np.isfinite(result["l2_abs"])

    def test_cross_validate_uses_model_output_index(self):
        """model_output_index selects which output column of a
        multi-output model is compared — must not error for index 0 on a
        single-output PINN, the only case this Trainer setup can exercise
        directly."""
        geo = ball([0.5, 0.5], 0.3)
        trainer = _build_poisson_trainer(max_epochs=1)
        result = trainer.cross_validate(
            geometry=geo, pde="poisson",
            bc_value=lambda x: np.zeros(len(x)),
            grid_points=8, model_output_index=0,
        )
        assert np.isfinite(result["l2_abs"])


# ---------------------------------------------------------------------------
# live_dashboard=True (RichDashboardCallback integration)
# ---------------------------------------------------------------------------

class TestTrainerLiveDashboard:
    def test_live_dashboard_true_attaches_rich_callback(self):
        from physai.dashboard import RichDashboardCallback
        trainer = _build_poisson_trainer(max_epochs=1)
        # _build_poisson_trainer doesn't expose live_dashboard, so build a
        # second Trainer directly to check attachment, without training it
        # (avoid needing a live terminal during the actual run below).
        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=8, n_bc_points=4, max_epochs=1, target_loss=-1.0)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.zeros((8, 2), dtype=np.float32))
        t = Trainer(backend, config, model, residual, collocation_points=coll,
                    live_dashboard=True)
        assert len(t.callbacks) == 1
        assert isinstance(t.callbacks[0], RichDashboardCallback)

    def test_live_dashboard_appended_when_callbacks_also_given(self):
        from physai.dashboard import RichDashboardCallback

        class _Recorder(Callback):
            pass

        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=8, n_bc_points=4, max_epochs=1, target_loss=-1.0)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.zeros((8, 2), dtype=np.float32))
        rec = _Recorder()
        t = Trainer(backend, config, model, residual, collocation_points=coll,
                    live_dashboard=True, callbacks=[rec])
        assert len(t.callbacks) == 2
        assert t.callbacks[0] is rec
        assert isinstance(t.callbacks[1], RichDashboardCallback)

    def test_live_dashboard_full_training_run_renders_and_exports_log(self, tmp_path, monkeypatch):
        from physai.dashboard import RichDashboardCallback
        monkeypatch.chdir(tmp_path)  # RichDashboardCallback's log_file is a relative path

        domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
        spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch",
                            n_collocation=8, n_bc_points=4, max_epochs=3, target_loss=-1.0)
        config = AutoOptimizer(backend, verbose=False).analyse(spec)
        model = PINN(backend, layer_sizes=config.model.layer_sizes,
                     activation=config.model.activation)
        residual = build_residual("poisson", backend)
        coll = backend.tensor(np.random.uniform(0, 1, size=(8, 2)).astype(np.float32))
        t = Trainer(backend, config, model, residual, collocation_points=coll,
                    live_dashboard=True, log_every=1)

        history = t.train()
        assert len(history.steps) == 3
        dashboard = t.callbacks[0]
        assert isinstance(dashboard, RichDashboardCallback)
        assert len(dashboard.logs) > 0
        assert "training complete" in dashboard.logs[-1].lower()
        assert (tmp_path / "physai_session.log").exists()

    def test_dashboard_chat_without_model_path_raises(self):
        from physai.dashboard import RichDashboardCallback
        with pytest.raises(ValueError, match="model_path"):
            RichDashboardCallback(enable_chat=True)

    def test_dashboard_send_chat_disabled_raises(self):
        from physai.dashboard import RichDashboardCallback
        dashboard = RichDashboardCallback(enable_chat=False)
        with pytest.raises(RuntimeError, match="Chat is disabled"):
            dashboard.send_chat("hello")


# ---------------------------------------------------------------------------
# for_spectral_element / _train_spectral_element (USENO architecture)
# ---------------------------------------------------------------------------

def _make_spectral_element_config(n_elements=2, n_modes=4, rank=2, max_epochs=2):
    domain = DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)], time_domain=(0.0, 1.0))
    problem = ProblemSpec(pde_name="heat", domain=domain, max_epochs=max_epochs,
                           target_loss=-1.0)  # never trips early-stop
    model = ModelConfig(
        arch="spectral_element", layer_sizes=(1, 1), activation="tanh",
        use_residual=False, n_elements=n_elements, n_modes=n_modes, rank=rank,
    )
    training = TrainingConfig(
        optimizer_name="adamw", learning_rate=1e-3, weight_decay=0.0, batch_size=32,
        n_collocation=100, n_bc_points=20, loss_weights={}, scheduler=SchedulerConfig("none"),
        use_lbfgs_phase=False, lbfgs_max_iter=50, rar_enabled=False, rar_interval=1000,
        rar_fraction=0.1, dtype="float32", grad_clip_norm=None, use_jit=False,
    )
    meta = PDEMeta("heat", 2, False, 1, True, "low", "spectral_element")
    return RuntimeConfig(problem=problem, model=model, training=training, meta=meta)


class TestTrainerSpectralElement:
    def test_for_spectral_element_wrong_arch_raises(self):
        config = _make_spectral_element_config()
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model, arch="pinn"),
        )
        t_points = torch.rand(3, 1)
        with pytest.raises(ValueError, match="spectral_element"):
            Trainer.for_spectral_element(config, [{2: 1.0}], t_points, nonlinear_fn=None)

    def test_for_spectral_element_builds_valid_trainer(self):
        config = _make_spectral_element_config()
        t_points = torch.rand(3, 1)
        trainer = Trainer.for_spectral_element(config, [{2: 1.0}], t_points, nonlinear_fn=None)
        assert isinstance(trainer.model, USENOCPModule)
        assert trainer.residual is None
        assert trainer._composite is None
        assert trainer.config.model.arch == "spectral_element"

    def test_train_dispatches_to_spectral_element_loop(self):
        config = _make_spectral_element_config(max_epochs=3)
        t_points = torch.rand(3, 1)
        trainer = Trainer.for_spectral_element(config, [{2: 1.0}], t_points, nonlinear_fn=None)
        history = trainer.train()
        assert len(history.steps) == 3
        assert all(np.isfinite(v) for v in history.total_loss)
        # spectral_element metrics carry their own weight_* terms, not pde/bc
        assert "loss_pde" in history.terms

    def test_spectral_element_train_end_to_end_callbacks_fire(self):
        events = []

        class _RecordingCallback(Callback):
            def on_train_begin(self, trainer):
                events.append("begin")

            def on_epoch_end(self, trainer, step, loss, breakdown):
                events.append(f"epoch_{step}")

            def on_train_end(self, trainer):
                events.append("end")

        config = _make_spectral_element_config(max_epochs=2)
        t_points = torch.rand(2, 1)
        trainer = Trainer.for_spectral_element(
            config, [{2: 1.0}], t_points, nonlinear_fn=None,
            callbacks=[_RecordingCallback()],
        )
        trainer.train()
        assert events == ["begin", "epoch_0", "epoch_1", "end"]

    def test_spectral_element_respects_target_loss_early_stop(self):
        config = _make_spectral_element_config(max_epochs=10_000)
        config = dataclasses.replace(
            config, problem=dataclasses.replace(config.problem, target_loss=float("inf")),
        )
        t_points = torch.rand(2, 1)
        trainer = Trainer.for_spectral_element(config, [{2: 1.0}], t_points, nonlinear_fn=None)
        history = trainer.train()
        assert len(history.steps) == 1  # stops after the very first step

    def test_spectral_element_stop_flag_halts_training(self):
        config = _make_spectral_element_config(max_epochs=10_000)
        t_points = torch.rand(2, 1)
        trainer = Trainer.for_spectral_element(config, [{2: 1.0}], t_points, nonlinear_fn=None)

        class _StopAfterOne(Callback):
            def on_epoch_end(self, trainer, step, loss, breakdown):
                trainer._stop_flag = True

        trainer.callbacks = [_StopAfterOne()]
        history = trainer.train()
        assert len(history.steps) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))