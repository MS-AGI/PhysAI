"""
tests/test_pinn.py

Tests for physai.models.pinn: FourierEmbedding, DirichletConstraint,
OutputScaler, the PINN wrapper itself, and the build_pinn factory.

Uses the TorchBackend only (PINN's forward() dispatches on backend.name,
but the torch path is representative and this library's other model
tests follow the same single-backend convention).
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from physai.backends.torch_backend import TorchBackend
from physai.models.pinn import (
    PINN,
    build_pinn,
    FourierEmbedding,
    DirichletConstraint,
    OutputScaler,
)

backend = TorchBackend(device="cpu")


def _points(n: int, cols: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-1.0, 1.0, size=(n, cols)).astype(np.float32)
    return backend.tensor(arr)


# ---------------------------------------------------------------------------
# FourierEmbedding
# ---------------------------------------------------------------------------

class TestFourierEmbedding:
    def test_output_dim_is_twice_n_frequencies(self):
        emb = FourierEmbedding(backend, n_input=2, n_frequencies=16)
        assert emb.output_dim == 32

    def test_call_output_shape(self):
        emb = FourierEmbedding(backend, n_input=2, n_frequencies=16)
        x = _points(5, 2)
        out = emb(x)
        assert out.shape == (5, 32)

    def test_output_is_bounded_sin_cos(self):
        """sin/cos features must lie in [-1, 1]."""
        emb = FourierEmbedding(backend, n_input=2, n_frequencies=8)
        x = _points(10, 2)
        out = emb(x)
        assert torch.all(out <= 1.0 + 1e-5) and torch.all(out >= -1.0 - 1e-5)

    def test_deterministic_B_matrix_across_instances(self):
        """_init_B always seeds with 42, so two embeddings with the same
        shape must produce bit-identical frequency matrices and outputs."""
        emb1 = FourierEmbedding(backend, n_input=3, n_frequencies=10)
        emb2 = FourierEmbedding(backend, n_input=3, n_frequencies=10)
        x = _points(4, 3)
        torch.testing.assert_close(emb1(x), emb2(x))

    def test_different_n_input_changes_B_shape(self):
        emb = FourierEmbedding(backend, n_input=5, n_frequencies=12)
        assert emb._B.shape == (5, 12)

    def test_first_half_is_sin_second_half_is_cos(self):
        emb = FourierEmbedding(backend, n_input=1, n_frequencies=4)
        x = backend.tensor([[0.0]])  # sin(0)=0, cos(0)=1
        out = emb(x)
        np.testing.assert_allclose(out[0, :4].numpy(), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(out[0, 4:].numpy(), np.ones(4), atol=1e-6)


# ---------------------------------------------------------------------------
# DirichletConstraint
# ---------------------------------------------------------------------------

class TestDirichletConstraint:
    def test_apply_formula(self):
        distance_fn = lambda x: x[..., 0]  # noqa: E731  (zero at x=0)
        boundary_fn = lambda x: (2.0 * torch.ones_like(x[..., 0]))[..., None]  # noqa: E731
        constraint = DirichletConstraint(distance_fn, boundary_fn)

        x = backend.tensor([[0.0], [1.0], [2.0]])
        raw = backend.tensor([[10.0], [10.0], [10.0]])
        out = constraint.apply(x, raw)
        # D*raw + g = [0*10+2, 1*10+2, 2*10+2]
        np.testing.assert_allclose(out.numpy().ravel(), [2.0, 12.0, 22.0])

    def test_zero_at_boundary_recovers_boundary_value(self):
        """Where distance_fn(x)=0, the raw network output must be fully
        overridden by the boundary function regardless of its value."""
        distance_fn = lambda x: torch.zeros_like(x[..., 0])  # noqa: E731
        boundary_fn = lambda x: (x[..., 0] ** 2)[..., None]  # noqa: E731
        constraint = DirichletConstraint(distance_fn, boundary_fn)

        x = backend.tensor([[3.0], [-2.0]])
        raw = backend.tensor([[999.0], [-999.0]])
        out = constraint.apply(x, raw)
        np.testing.assert_allclose(out.numpy().ravel(), [9.0, 4.0])


# ---------------------------------------------------------------------------
# OutputScaler
# ---------------------------------------------------------------------------

class TestOutputScaler:
    def test_affine_rescaling(self):
        scaler = OutputScaler(backend, scales=[2.0, 0.5], shifts=[1.0, -1.0])
        out = backend.tensor([[1.0, 4.0], [2.0, 8.0]])
        scaled = scaler.apply(out)
        expected = np.array([[2.0 * 1.0 + 1.0, 0.5 * 4.0 - 1.0],
                              [2.0 * 2.0 + 1.0, 0.5 * 8.0 - 1.0]])
        np.testing.assert_allclose(scaled.numpy(), expected)

    def test_default_shifts_are_zero(self):
        scaler = OutputScaler(backend, scales=[3.0])
        out = backend.tensor([[2.0]])
        np.testing.assert_allclose(scaler.apply(out).numpy(), [[6.0]])


# ---------------------------------------------------------------------------
# PINN — construction and forward
# ---------------------------------------------------------------------------

class TestPINNConstruction:
    def test_forward_output_shape(self):
        pinn = PINN(backend, layer_sizes=(2, 16, 16, 1), activation="tanh")
        x = _points(7, 2)
        out = pinn.forward(x)
        assert out.shape == (7, 1)

    def test_model_fn_matches_forward(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        x = _points(4, 2)
        torch.testing.assert_close(pinn.model_fn(x), pinn.forward(x))

    def test_use_fourier_replaces_first_layer_size(self):
        """With use_fourier=True, the embedding's output_dim (not the raw
        n_input) must feed the first Linear layer."""
        pinn = PINN(
            backend, layer_sizes=(2, 16, 1), activation="tanh",
            use_fourier=True, fourier_freqs=10,
        )
        assert pinn.embedding is not None
        assert pinn.embedding.output_dim == 20
        first_linear = next(m for m in pinn._model.modules() if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == 20

        x = _points(5, 2)
        out = pinn.forward(x)
        assert out.shape == (5, 1)

    def test_no_fourier_embedding_is_none(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        assert pinn.embedding is None

    def test_use_residual_forwarded_to_build_mlp(self):
        pinn = PINN(backend, layer_sizes=(4, 4, 4, 1), activation="relu", use_residual=True)
        from physai.backends.torch_backend import _ResidualLinear
        residual_layers = [m for m in pinn._model.net if isinstance(m, _ResidualLinear)]
        assert len(residual_layers) > 0

    def test_dirichlet_constraint_applied_in_forward(self):
        distance_fn = lambda x: torch.zeros_like(x[..., 0])  # noqa: E731  (always on boundary)
        boundary_fn = lambda x: (5.0 * torch.ones_like(x[..., 0]))[..., None]  # noqa: E731
        constraint = DirichletConstraint(distance_fn, boundary_fn)

        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh", constraint=constraint)
        x = _points(6, 2)
        out = pinn.forward(x)
        # distance is identically zero everywhere, so output must be
        # exactly the boundary value regardless of the raw network output.
        np.testing.assert_allclose(out.detach().numpy().ravel(), [5.0] * 6)

    def test_output_scaler_applied_in_forward(self):
        scaler = OutputScaler(backend, scales=[0.0], shifts=[7.0])
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh", output_scaler=scaler)
        x = _points(6, 2)
        out = pinn.forward(x)
        # scale=0 zeroes out whatever the network produced, leaving only the shift.
        np.testing.assert_allclose(out.detach().numpy().ravel(), [7.0] * 6)

    def test_dtype_override(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh", dtype=torch.float64)
        first_linear = next(m for m in pinn._model.modules() if isinstance(m, torch.nn.Linear))
        assert first_linear.weight.dtype == torch.float64


# ---------------------------------------------------------------------------
# PINN — parameters / training-adjacent interface
# ---------------------------------------------------------------------------

class TestPINNParameterInterface:
    def test_parameters_nonempty_and_tensors(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        params = pinn.parameters()
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_named_parameters_is_dict_of_tensors(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        named = pinn.named_parameters()
        assert isinstance(named, dict)
        assert len(named) > 0
        assert all(isinstance(v, torch.Tensor) for v in named.values())

    def test_zero_grad_clears_gradients(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        x = _points(4, 2)
        loss = pinn.forward(x).sum()
        loss.backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in pinn.parameters())
        pinn.zero_grad()
        for p in pinn.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_repr_contains_backend_and_param_count(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        r = repr(pinn)
        assert "torch" in r
        assert "n_params" in r


# ---------------------------------------------------------------------------
# PINN — save/load
# ---------------------------------------------------------------------------

class TestPINNSave:
    def test_save_writes_npz_with_matching_shapes(self):
        pinn = PINN(backend, layer_sizes=(2, 8, 1), activation="tanh")
        named = pinn.named_parameters()

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pinn_weights.npz")
            pinn.save(path)
            assert os.path.exists(path)
            with np.load(path) as loaded:
                assert set(loaded.files) == set(named.keys())
                for k, v in named.items():
                    np.testing.assert_allclose(loaded[k], v.detach().numpy())


# ---------------------------------------------------------------------------
# build_pinn factory
# ---------------------------------------------------------------------------

class TestBuildPinnFactory:
    def test_factory_produces_correct_layer_sizes(self):
        pinn = build_pinn(backend, n_input=3, n_output=2, hidden_sizes=(16, 16))
        linears = [m for m in pinn._model.modules() if isinstance(m, torch.nn.Linear)]
        assert linears[0].in_features == 3
        assert linears[-1].out_features == 2

    def test_factory_forward_shape(self):
        pinn = build_pinn(backend, n_input=2, n_output=3, hidden_sizes=(8, 8))
        x = _points(5, 2)
        out = pinn.model_fn(x)
        assert out.shape == (5, 3)

    def test_factory_with_fourier(self):
        pinn = build_pinn(
            backend, n_input=2, n_output=1, hidden_sizes=(16,),
            use_fourier=True, fourier_freqs=8,
        )
        assert pinn.embedding is not None
        x = _points(3, 2)
        out = pinn.model_fn(x)
        assert out.shape == (3, 1)

    def test_factory_passes_through_constraint_and_scaler(self):
        distance_fn = lambda x: torch.zeros_like(x[..., 0])  # noqa: E731
        boundary_fn = lambda x: torch.ones_like(x[..., 0:1])  # noqa: E731
        constraint = DirichletConstraint(distance_fn, boundary_fn)
        scaler = OutputScaler(backend, scales=[1.0], shifts=[2.0])

        pinn = build_pinn(
            backend, n_input=2, n_output=1, hidden_sizes=(8,),
            constraint=constraint, output_scaler=scaler,
        )
        x = _points(4, 2)
        out = pinn.model_fn(x)
        # constraint forces raw output to exactly 1.0, then the scaler maps
        # 1.0 -> 1.0*1.0 + 2.0 = 3.0
        np.testing.assert_allclose(out.detach().numpy().ravel(), [3.0] * 4)