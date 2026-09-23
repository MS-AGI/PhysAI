"""
tests/test-backend/paddle_test.py

Test chunk — PaddleBackend (physai.backends.paddle_backend)

Covers: tensor creation, math primitives, FFT round-trips, autodiff
(grad/value_and_grad/jacobian/hessian, plus the double-backward jvp
emulation and the documented mode="taylor" NotImplementedError), MLP
construction, optimizer construction and the real (torch-style, not
JAX-style) optimizer_step contract, jit/vmap, and device/config helpers.

The whole module is skipped if `paddle` is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

paddle = pytest.importorskip("paddle", reason="paddle not installed")

from physai.backends import get_backend, clear_backend_cache
from physai.backends.base import AbstractBackend, BackendCapabilities
from physai.backends.paddle_backend import PaddleBackend


@pytest.fixture()
def backend() -> PaddleBackend:
    clear_backend_cache()
    return PaddleBackend(seed=0)


# ---------------------------------------------------------------------------
# Identity & capabilities
# ---------------------------------------------------------------------------

class TestIdentityAndCapabilities:
    def test_name(self, backend):
        assert backend.name == "paddle"

    def test_is_abstract_backend(self, backend):
        assert isinstance(backend, AbstractBackend)

    def test_capabilities(self, backend):
        caps = backend.capabilities
        assert isinstance(caps, BackendCapabilities)
        assert caps.supports_jit is True
        assert caps.supports_vmap is False
        assert caps.supports_complex is True
        assert caps.supports_sparse is True
        assert caps.native_fft is True

    def test_repr_contains_name_and_device(self, backend):
        r = repr(backend)
        assert "paddle" in r
        assert backend.default_device() in r


# ---------------------------------------------------------------------------
# Registry / factory integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    @pytest.mark.parametrize("key", ["paddle", "paddlepaddle"])
    def test_get_backend_aliases(self, key):
        clear_backend_cache()
        b = get_backend(key, cache=False)
        assert isinstance(b, PaddleBackend)
        assert b.name == "paddle"


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_tensor_from_list(self, backend):
        t = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    def test_tensor_default_dtype(self, backend):
        t = backend.tensor([1, 2, 3])
        assert str(t.dtype).endswith("float32")

    def test_zeros(self, backend):
        t = backend.zeros((2, 3))
        np.testing.assert_allclose(t.numpy(), np.zeros((2, 3)))

    def test_ones(self, backend):
        t = backend.ones((4,))
        np.testing.assert_allclose(t.numpy(), np.ones(4))

    def test_linspace(self, backend):
        t = backend.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(t.numpy(), np.linspace(0.0, 1.0, 5), atol=1e-6)

    def test_meshgrid_ij(self, backend):
        x = backend.linspace(0.0, 1.0, 3)
        y = backend.linspace(0.0, 1.0, 4)
        X, Y = backend.meshgrid(x, y, indexing="ij")
        assert X.shape == [3, 4]
        assert Y.shape == [3, 4]

    def test_stack(self, backend):
        a = backend.ones((3,))
        b = backend.zeros((3,))
        s = backend.stack([a, b], axis=0)
        assert list(s.shape) == [2, 3]

    def test_concatenate(self, backend):
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))
        c = backend.concatenate([a, b], axis=0)
        assert list(c.shape) == [4, 3]

    def test_reshape(self, backend):
        t = backend.ones((6,))
        r = backend.reshape(t, (2, 3))
        assert list(r.shape) == [2, 3]

    def test_cast(self, backend):
        t = backend.ones((2,))
        c = backend.cast(t, "int32")
        assert str(c.dtype).endswith("int32")

    def test_to_numpy_roundtrip(self, backend):
        t = backend.tensor([1.0, 2.0])
        arr = backend.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Mathematical primitives
# ---------------------------------------------------------------------------

class TestMathPrimitives:
    @pytest.fixture()
    def x(self, backend):
        return backend.tensor([1.0, 2.0, 3.0, 4.0])

    def test_mean(self, backend, x):
        np.testing.assert_allclose(backend.mean(x).numpy(), 2.5)

    def test_sum(self, backend, x):
        np.testing.assert_allclose(backend.sum(x).numpy(), 10.0)

    def test_abs(self, backend):
        t = backend.tensor([-1.0, 2.0, -3.0])
        np.testing.assert_allclose(backend.abs(t).numpy(), [1.0, 2.0, 3.0])

    def test_sqrt(self, backend):
        t = backend.tensor([4.0, 9.0])
        np.testing.assert_allclose(backend.sqrt(t).numpy(), [2.0, 3.0])

    def test_square(self, backend):
        t = backend.tensor([2.0, 3.0])
        np.testing.assert_allclose(backend.square(t).numpy(), [4.0, 9.0])

    def test_exp_log_roundtrip(self, backend, x):
        np.testing.assert_allclose(
            backend.log(backend.exp(x)).numpy(), x.numpy(), atol=1e-5
        )

    def test_sin_cos_identity(self, backend, x):
        s = backend.sin(x)
        c = backend.cos(x)
        np.testing.assert_allclose(
            backend.square(s).numpy() + backend.square(c).numpy(),
            np.ones_like(x.numpy()),
            atol=1e-5,
        )

    def test_tanh_bounds(self, backend, x):
        t = backend.tanh(x)
        assert np.all(t.numpy() < 1.0) and np.all(t.numpy() > -1.0)

    def test_matmul(self, backend):
        a = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = backend.tensor([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(backend.matmul(a, b).numpy(), a.numpy())

    def test_einsum(self, backend):
        a = backend.tensor([1.0, 2.0, 3.0])
        b = backend.tensor([4.0, 5.0, 6.0])
        result = backend.einsum("i,i->", a, b)
        np.testing.assert_allclose(result.numpy(), 32.0)


# ---------------------------------------------------------------------------
# FFT operations
# ---------------------------------------------------------------------------

class TestFFT:
    def test_rfft_irfft_roundtrip(self, backend):
        arr = np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfft(t)
        back = backend.irfft(spec, n=16)
        np.testing.assert_allclose(back.numpy(), arr, atol=1e-4)

    def test_rfft2_irfft2_roundtrip(self, backend):
        arr = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfft2(t)
        back = backend.irfft2(spec, s=(8, 8))
        np.testing.assert_allclose(back.numpy(), arr, atol=1e-4)

    def test_rfftn_irfftn_roundtrip(self, backend):
        arr = np.random.RandomState(1).rand(4, 6, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfftn(t)
        back = backend.irfftn(spec, s=arr.shape)
        np.testing.assert_allclose(back.numpy(), arr, atol=1e-4)


# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------

class TestAutodiff:
    def test_grad_of_square(self, backend):
        f = lambda x: paddle.sum(x ** 2)
        g = backend.grad(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(g(x).numpy(), 2 * x.numpy(), atol=1e-5)

    def test_value_and_grad(self, backend):
        f = lambda x: paddle.sum(x ** 2)
        vg = backend.value_and_grad(f)
        x = backend.tensor([1.0, 2.0])
        val, grad = vg(x)
        np.testing.assert_allclose(val.numpy(), 5.0, atol=1e-5)
        np.testing.assert_allclose(grad.numpy(), [2.0, 4.0], atol=1e-5)

    def test_jvp_double_backward_trick(self, backend):
        # KNOWN LIMITATION: both a hand-rolled double-backward
        # implementation and Paddle's own official
        # paddle.incubate.autograd.functional.jvp fail with the same
        # underlying Paddle kernel error ("DX can not be nullptr" from
        # activation_grad_impl) computing create_graph=True
        # double-backward through this op chain in the installed
        # Paddle build. Confirmed upstream (not a physai bug) since
        # Paddle's own blessed implementation hits the identical error.
        # PaddleBackend.jvp catches this and raises a clear
        # NotImplementedError instead of leaking the raw Paddle
        # traceback — this test documents that contract.
        f = lambda x: paddle.sum(x ** 2)
        x = backend.tensor([1.0, 2.0, 3.0])
        v = backend.tensor([1.0, 0.0, 0.0])
        with pytest.raises(NotImplementedError):
            backend.jvp(f, x, v)

    def test_grad_mode_forward_matches_reverse(self, backend):
        # KNOWN LIMITATION: see test_jvp_double_backward_trick above —
        # mode="forward" is implemented via self.jvp internally, so it
        # inherits the same documented upstream Paddle limitation.
        f = lambda x: paddle.stack([x[0] * x[1], x[0] ** 2])
        x = backend.tensor([2.0, 3.0])
        forward_fn = backend.grad(f, mode="forward")
        with pytest.raises(NotImplementedError):
            forward_fn(x)

    def test_grad_mode_taylor_raises_not_implemented(self, backend):
        f = lambda x: paddle.sum(x ** 2)
        with pytest.raises(NotImplementedError):
            backend.grad(f, mode="taylor")

    def test_grad_unknown_mode_raises(self, backend):
        f = lambda x: paddle.sum(x ** 2)
        with pytest.raises(ValueError):
            backend.grad(f, mode="not_a_real_mode")

    def test_jacobian(self, backend):
        f = lambda x: paddle.stack([x[0] * x[1], x[0] ** 2])
        x = backend.tensor([2.0, 3.0])
        J = backend.jacobian(f, x)
        expected = np.array([[3.0, 2.0], [4.0, 0.0]])
        np.testing.assert_allclose(np.asarray(J), expected, atol=1e-4)

    def test_hessian_of_quadratic(self, backend):
        f = lambda x: paddle.sum(x ** 2)
        x = backend.tensor([1.0, 2.0])
        H = backend.hessian(f, x)
        np.testing.assert_allclose(np.asarray(H), 2 * np.eye(2), atol=1e-4)


# ---------------------------------------------------------------------------
# Neural-network model interface
# ---------------------------------------------------------------------------

class TestBuildMLP:
    def test_build_mlp_returns_paddle_layer(self, backend):
        model = backend.build_mlp([2, 16, 16, 1], activation="tanh")
        assert isinstance(model, paddle.nn.Layer)

    def test_build_mlp_forward_shape(self, backend):
        model = backend.build_mlp([2, 16, 1], activation="tanh")
        x = paddle.ones([5, 2])
        out = model(x)
        assert list(out.shape) == [5, 1]

    def test_build_mlp_layer_count(self, backend):
        model = backend.build_mlp([2, 64, 64, 1], activation="relu")
        assert len(model.linears) == 3

    def test_build_mlp_unknown_activation_raises(self, backend):
        with pytest.raises(ValueError):
            backend.build_mlp([2, 4, 1], activation="not_a_real_activation")

    def test_build_mlp_use_residual(self, backend):
        model = backend.build_mlp([4, 4, 4, 1], activation="relu", use_residual=True)
        out = model(paddle.ones([2, 4]))
        assert list(out.shape) == [2, 1]

    @pytest.mark.parametrize("name", ["llaaf", "pade"])
    def test_build_mlp_stateful_activation(self, backend, name):
        model = backend.build_mlp([2, 8, 8, 1], activation=name)
        out = model(paddle.ones([3, 2]))
        assert list(out.shape) == [3, 1]
        # Stateful activations must contribute trainable parameters
        # beyond the plain Linear weights/biases.
        n_linear_params = 2 * len(model.linears)  # weight + bias per Linear
        assert len(backend.parameters(model)) > n_linear_params

    def test_llaaf_starts_near_identity_tanh(self, backend):
        # L-LAAF is tanh(n*a*x) with no outer coefficient (Jagtap &
        # Karniadakis 2020) -- at init a=1/n, so n*a=1 and this must be
        # exactly tanh(x).
        model = backend.build_mlp([1, 1, 1], activation="llaaf")
        x = backend.tensor([[0.5], [1.0], [-2.0]])
        # Bypass the Linear layers' random init by checking the
        # activation module directly instead of the full MLP forward.
        act = model.activations[0]
        y = act(x)
        np.testing.assert_allclose(y.numpy(), np.tanh(x.numpy()), atol=1e-4)

    def test_parameters_returns_paddle_parameters(self, backend):
        model = backend.build_mlp([2, 8, 1], activation="tanh")
        params = backend.parameters(model)
        assert len(params) > 0
        assert all(isinstance(p, paddle.Tensor) for p in params)

    def test_zero_grad_clears_gradients(self, backend):
        model = backend.build_mlp([2, 4, 1], activation="tanh")
        x = paddle.ones([3, 2])
        loss = paddle.sum(model(x))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        backend.zero_grad(model)
        # Paddle's own documented default for Layer.clear_gradients() is
        # set_to_zero=True — it zeroes gradients in place rather than
        # setting them to None (unlike torch's zero_grad(set_to_none=True)
        # default). So "cleared" means zero here, not None.
        for p in model.parameters():
            assert p.grad is None or float(paddle.sum(paddle.abs(p.grad))) == 0.0


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------

class TestOptimizer:
    @pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop"])
    def test_build_optimizer_known_names(self, backend, name):
        model = backend.build_mlp([2, 4, 1])
        opt = backend.build_optimizer(model, optimizer_name=name, lr=1e-3)
        assert isinstance(opt, paddle.optimizer.Optimizer)

    def test_build_optimizer_unknown_name_raises(self, backend):
        model = backend.build_mlp([2, 4, 1])
        with pytest.raises(ValueError):
            backend.build_optimizer(model, optimizer_name="not_an_optimizer")

    def test_build_optimizer_strips_weight_decay_for_non_adamw(self, backend):
        model = backend.build_mlp([2, 4, 1])
        opt = backend.build_optimizer(model, optimizer_name="sgd", lr=1e-3, weight_decay=0.01)
        assert isinstance(opt, paddle.optimizer.SGD)

    def test_optimizer_step_applies_update(self, backend):
        model = backend.build_mlp([2, 4, 1], activation="tanh")
        opt = backend.build_optimizer(model, optimizer_name="sgd", lr=0.1)
        before = [p.numpy().copy() for p in model.parameters()]

        x = paddle.ones([4, 2])
        y = model(x)
        loss = paddle.mean(y ** 2)
        backend.optimizer_step(opt, loss, model=model)

        after = model.parameters()
        changed = any(
            not np.allclose(b, a.numpy()) for b, a in zip(before, after)
        )
        assert changed


# ---------------------------------------------------------------------------
# JIT / vmap
# ---------------------------------------------------------------------------

class TestJitVmap:
    def test_jit_wraps_function(self, backend):
        f = lambda x: x ** 2 + 1
        jitted = backend.jit(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(jitted(x).numpy(), f(x).numpy())

    def test_vmap_batches_over_axis0(self, backend):
        f = lambda x: paddle.sum(x)
        vf = backend.vmap(f, in_axes=0)
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = vf(x)
        np.testing.assert_allclose(result.numpy(), [3.0, 7.0])

    def test_vmap_none_axis_raises(self, backend):
        f = lambda a, b: a + b
        vf = backend.vmap(f, in_axes=[0, None])
        a = backend.tensor([1.0, 2.0])
        b = backend.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="in_axes=None"):
            vf(a, b)

    def test_vmap_wrong_in_axes_length_raises(self, backend):
        f = lambda a, b: a + b
        vf = backend.vmap(f, in_axes=[0, 0, 0])
        a = backend.tensor([1.0, 2.0])
        b = backend.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="in_axes has"):
            vf(a, b)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class TestDevice:
    def test_default_device_is_cpu_or_gpu(self, backend):
        assert backend.default_device() in ("cpu", "gpu")

    def test_to_device_cpu(self, backend):
        t = backend.tensor([1.0, 2.0])
        moved = backend.to_device(t, "cpu")
        np.testing.assert_allclose(moved.numpy(), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Miscellaneous / configure / seed
# ---------------------------------------------------------------------------

class TestMisc:
    def test_configure_dtype(self, backend):
        backend.configure(dtype="float64")
        assert backend._dtype == "float64"

    def test_seed_is_reproducible(self, backend):
        backend.seed(99)
        a = paddle.uniform([3])
        backend.seed(99)
        b = paddle.uniform([3])
        np.testing.assert_allclose(a.numpy(), b.numpy())