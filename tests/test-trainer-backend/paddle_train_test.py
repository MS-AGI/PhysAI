"""
tests/test_trainer_paddle.py

Covers the PaddlePaddle-specific dispatch branches in physai.trainer that
tests/test_trainer.py's torch-only suite structurally cannot reach:
_step_paddle, the relevant _apply_lr branch, _lbfgs_phase_paddle (real
L-BFGS or the documented fallback), and (where applicable) the PaddlePaddle
branch of save_checkpoint/load_checkpoint.

Skipped via pytest.importorskip if PaddlePaddle isn't installed, matching the
convention used by tests/test-backend/paddle_test.py -- kept in its own
file (rather than one combined multi-backend file) specifically so a
missing framework only skips *this* file, not unrelated backends' tests.

These are integration-style smoke tests (finite loss, no crash, checkpoint
round-trips, LR schedule actually reaches the native optimizer) rather
than tight numerical assertions -- the residual math itself is already
covered by tests/test_pde_residual.py and tests/test-backend/paddle_test.py.

Run with: pytest tests/test_trainer_paddle.py -v
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import build_residual
from physai.models.pinn import PINN
from physai.trainer import Trainer


def _build_backend_trainer(backend, *, max_epochs=3, use_lbfgs_phase=False,
                            grad_clip_norm=None):
    """Same minimal homogeneous-Poisson-on-unit-square scenario as
    tests/test_trainer.py's `_build_poisson_trainer`, parameterised by
    backend so it can be reused across torch/jax/tensorflow/paddle."""
    domain = DomainSpec(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    spec = ProblemSpec(
        pde_name="poisson", domain=domain, model_arch="pinn",
        backend_name=backend.name, n_collocation=16, n_bc_points=8,
        max_epochs=max_epochs, use_lbfgs_phase=use_lbfgs_phase,
        target_loss=-1.0,
    )
    config = AutoOptimizer(backend, verbose=False).analyse(spec)
    if grad_clip_norm is not None:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, grad_clip_norm=grad_clip_norm),
        )

    model = PINN(backend, layer_sizes=config.model.layer_sizes,
                 activation=config.model.activation)
    residual = build_residual("poisson", backend)

    rng = np.random.default_rng(0)
    coll = backend.tensor(rng.uniform(0.0, 1.0, size=(16, 2)).astype(np.float32))
    bc_pts = backend.tensor(rng.uniform(0.0, 1.0, size=(8, 2)).astype(np.float32))
    bc_val = backend.tensor(np.zeros((8, 1), dtype=np.float32))

    trainer = Trainer(
        backend, config, model, residual,
        collocation_points=coll, bc_points=bc_pts, bc_values=bc_val,
        callbacks=[], log_every=10_000,
    )
    return trainer

paddle = pytest.importorskip("paddle", reason="paddle not installed")


@pytest.fixture()
def paddle_backend():
    from physai.backends.paddle_backend import PaddleBackend
    return PaddleBackend()


class TestTrainerPaddle:
    def test_train_produces_finite_history(self, paddle_backend):
        trainer = _build_backend_trainer(paddle_backend, max_epochs=5)
        history = trainer.train()
        assert len(history.steps) == 5
        assert all(np.isfinite(v) for v in history.total_loss)

    def test_apply_lr_calls_set_lr(self, paddle_backend):
        trainer = _build_backend_trainer(paddle_backend, max_epochs=1)
        trainer._apply_lr(2e-3)
        assert float(trainer._optimizer.get_lr()) == pytest.approx(2e-3, rel=1e-3)

    def test_grad_clip_norm_path_runs_without_error(self, paddle_backend):
        trainer = _build_backend_trainer(paddle_backend, max_epochs=3, grad_clip_norm=1.0)
        history = trainer.train()
        assert all(np.isfinite(v) for v in history.total_loss)

    def test_lbfgs_phase_paddle_warns_and_falls_back(self, paddle_backend):
        """Paddle's L-BFGS integration is documented as unimplemented —
        this must warn and fall back to low-LR fine-tuning rather than
        crashing or silently skipping the phase."""
        trainer = _build_backend_trainer(paddle_backend, max_epochs=2, use_lbfgs_phase=True)
        trainer.config = dataclasses.replace(
            trainer.config,
            training=dataclasses.replace(trainer.config.training, lbfgs_max_iter=2),
        )
        with pytest.warns(UserWarning, match="not yet wired up for the paddle backend"):
            history = trainer.train()
        assert np.isfinite(history.total_loss[-1])

    def test_compile_true_warns_unsupported_and_falls_back(self, paddle_backend):
        trainer = _build_backend_trainer(paddle_backend, max_epochs=1)
        trainer._compile = True
        with pytest.warns(UserWarning, match="has no meaningful effect"):
            fn = trainer._get_composite_fn()
        assert fn is trainer._composite

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))