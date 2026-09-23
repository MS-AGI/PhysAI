"""
tests/test_trainer_jax.py

Covers the JAX-specific dispatch branches in physai.trainer that
tests/test_trainer.py's torch-only suite structurally cannot reach:
_step_jax, the relevant _apply_lr branch, _lbfgs_phase_jax (real
L-BFGS or the documented fallback), and (where applicable) the JAX
branch of save_checkpoint/load_checkpoint.

Skipped via pytest.importorskip if JAX isn't installed, matching the
convention used by tests/test-backend/jax_test.py -- kept in its own
file (rather than one combined multi-backend file) specifically so a
missing framework only skips *this* file, not unrelated backends' tests.

These are integration-style smoke tests (finite loss, no crash, checkpoint
round-trips, LR schedule actually reaches the native optimizer) rather
than tight numerical assertions -- the residual math itself is already
covered by tests/test_pde_residual.py and tests/test-backend/jax_test.py.

Run with: pytest tests/test_trainer_jax.py -v
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

jax = pytest.importorskip("jax", reason="jax not installed")


@pytest.fixture()
def jax_backend():
    from physai.backends.jax_backend import JAXBackend
    return JAXBackend()


class TestTrainerJAX:
    def test_step_jax_requires_init_jax_first(self, jax_backend):
        trainer = _build_backend_trainer(jax_backend, max_epochs=1)
        with pytest.raises(RuntimeError, match="init_jax"):
            trainer._single_step()

    def test_init_jax_then_train_produces_finite_history(self, jax_backend):
        trainer = _build_backend_trainer(jax_backend, max_epochs=5)
        dummy = jax_backend.tensor(np.zeros((1, 2), dtype=np.float32))
        trainer.init_jax(dummy)
        history = trainer.train()
        assert len(history.steps) == 5
        assert all(np.isfinite(v) for v in history.total_loss)

    def test_apply_lr_mutates_opt_state_hyperparams(self, jax_backend):
        trainer = _build_backend_trainer(jax_backend, max_epochs=1)
        dummy = jax_backend.tensor(np.zeros((1, 2), dtype=np.float32))
        trainer.init_jax(dummy)
        trainer._apply_lr(1e-2)
        assert trainer._jax_opt_state.hyperparams["learning_rate"] == pytest.approx(1e-2)

    def test_apply_lr_noop_before_init_jax(self, jax_backend):
        """No opt_state exists yet — must not raise, just do nothing."""
        trainer = _build_backend_trainer(jax_backend, max_epochs=1)
        trainer._apply_lr(1e-2)  # should not raise

    def test_grad_clip_norm_path_runs_without_error(self, jax_backend):
        trainer = _build_backend_trainer(jax_backend, max_epochs=3, grad_clip_norm=1.0)
        dummy = jax_backend.tensor(np.zeros((1, 2), dtype=np.float32))
        trainer.init_jax(dummy)
        history = trainer.train()
        assert all(np.isfinite(v) for v in history.total_loss)

    def test_checkpoint_round_trip(self, jax_backend, tmp_path):
        trainer = _build_backend_trainer(jax_backend, max_epochs=3)
        dummy = jax_backend.tensor(np.zeros((1, 2), dtype=np.float32))
        trainer.init_jax(dummy)
        trainer.train()

        ckpt = str(tmp_path / "jax_ckpt")
        trainer.save_checkpoint(ckpt)
        step_before = trainer._step
        trainer._step = 0
        trainer.load_checkpoint(ckpt)
        assert trainer._step == step_before
        # params successfully rebuilt into the same pytree structure
        leaves_before = jax.tree_util.tree_leaves(trainer._jax_params)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves_before)

    def test_lbfgs_phase_jax_runs_to_completion(self, jax_backend):
        trainer = _build_backend_trainer(jax_backend, max_epochs=2, use_lbfgs_phase=True)
        trainer.config = dataclasses.replace(
            trainer.config,
            training=dataclasses.replace(trainer.config.training, lbfgs_max_iter=2),
        )
        dummy = jax_backend.tensor(np.zeros((1, 2), dtype=np.float32))
        trainer.init_jax(dummy)
        history = trainer.train()  # runs main loop + _lbfgs_phase_jax
        assert np.isfinite(history.total_loss[-1])

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))