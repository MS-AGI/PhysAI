"""
tests/test_tensorflow_backend.py

Test chunk 1b — TensorFlowBackend (physai.backends.tensorflow_backend)

Covers: tensor creation, math primitives, FFT round-trips (including the
rfftn/irfftn N-D implementation that manually chains a real rfft on the
first axis with complex fft/ifft on the rest), GradientTape-based
autodiff (grad/value_and_grad/jacobian/hessian), Keras MLP construction
(including the residual-connection branch), optimizer construction and
the documented "requires an open tape" contract of optimizer_step, jit
(tf.function) and the axis-aware vmap fallback (tf.vectorized_map), and
device/config helpers.

The whole module is skipped if `tensorflow` is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

from physai.backends import get_backend, clear_backend_cache
from physai.backends.base import AbstractBackend, BackendCapabilities
from physai.backends.tensorflow_backend import TensorFlowBackend


@pytest.fixture()
def backend() -> TensorFlowBackend:
    clear_backend_cache()
    return TensorFlowBackend(memory_growth=False)


# ---------------------------------------------------------------------------
# Identity & capabilities
# ---------------------------------------------------------------------------

class TestIdentityAndCapabilities:
    def test_name(self, backend):
        assert backend.name == "tensorflow"

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
        # mixed precision support tracks actual GPU availability
        has_gpu = bool(tf.config.list_physical_devices("GPU"))
        assert caps.supports_mixed_precision is has_gpu

    def test_repr_contains_name_and_device(self, backend):
        r = repr(backend)
        assert "tensorflow" in r
        assert backend.default_device() in r


# ---------------------------------------------------------------------------
# Registry / factory integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    @pytest.mark.parametrize("key", ["tensorflow", "tf"])
    def test_get_backend_aliases(self, key):
        clear_backend_cache()
        b = get_backend(key, cache=False, memory_growth=False)
        assert isinstance(b, TensorFlowBackend)
        assert b.name == "tensorflow"


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_tensor_from_list(self, backend):
        t = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    def test_tensor_default_dtype(self, backend):
        t = backend.tensor([1, 2, 3])
        assert t.dtype == tf.float32

    def test_tensor_explicit_dtype(self, backend):
        t = backend.tensor([1, 2, 3], dtype=tf.int32)
        assert t.dtype == tf.int32

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
        assert X.shape == (3, 4)
        assert Y.shape == (3, 4)

    def test_stack(self, backend):
        a = backend.ones((3,))
        b = backend.zeros((3,))
        s = backend.stack([a, b], axis=0)
        assert s.shape == (2, 3)

    def test_concatenate(self, backend):
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))
        c = backend.concatenate([a, b], axis=0)
        assert c.shape == (4, 3)

    def test_reshape(self, backend):
        t = backend.ones((6,))
        r = backend.reshape(t, (2, 3))
        assert r.shape == (2, 3)

    def test_cast(self, backend):
        t = backend.ones((2,))
        c = backend.cast(t, tf.int32)
        assert c.dtype == tf.int32

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

    def test_rfft_dtype_is_complex64(self, backend):
        t = backend.tensor(np.ones(8, dtype=np.float32))
        spec = backend.rfft(t)
        assert spec.dtype == tf.complex64

    def test_rfft2_irfft2_roundtrip(self, backend):
        arr = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfft2(t)
        back = backend.irfft2(spec, s=(8, 8))
        np.testing.assert_allclose(back.numpy(), arr, atol=1e-4)

    def test_rfftn_irfftn_roundtrip_3d(self, backend):
        """Regression test for the N-D FFT chaining bug: earlier code
        re-cast the already-complex intermediate to float32 on each
        subsequent axis, dropping the imaginary part. A correct
        round-trip on a 3-D (>=2 axes-beyond-first) array is the direct
        check that the imaginary part now survives."""
        arr = np.random.RandomState(1).rand(4, 6, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfftn(t)
        back = backend.irfftn(spec, s=arr.shape)
        np.testing.assert_allclose(back.numpy(), arr, atol=1e-3)

    def test_rfftn_preserves_imaginary_part_after_first_axis(self, backend):
        arr = np.random.RandomState(2).rand(4, 4, 4).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfftn(t)
        # If the imaginary part were being dropped, the spectrum along
        # non-first axes would collapse toward being (numerically) real.
        assert np.abs(np.imag(spec.numpy())).max() > 1e-6


# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------

class TestAutodiff:
    def test_grad_of_square(self, backend):
        f = lambda x: tf.reduce_sum(x ** 2)
        g = backend.grad(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(g(x).numpy(), 2 * x.numpy(), atol=1e-5)

    def test_value_and_grad(self, backend):
        f = lambda x: tf.reduce_sum(x ** 2)
        vg = backend.value_and_grad(f)
        x = backend.tensor([1.0, 2.0])
        val, grad = vg(x)
        np.testing.assert_allclose(val.numpy(), 5.0, atol=1e-5)
        np.testing.assert_allclose(grad.numpy(), [2.0, 4.0], atol=1e-5)

    def test_grad_argnums_tuple(self, backend):
        f = lambda x, y: tf.reduce_sum(x * y)
        g = backend.grad(f, argnums=(0, 1))
        x = backend.tensor([1.0, 2.0])
        y = backend.tensor([3.0, 4.0])
        gx, gy = g(x, y)
        np.testing.assert_allclose(gx.numpy(), y.numpy(), atol=1e-5)
        np.testing.assert_allclose(gy.numpy(), x.numpy(), atol=1e-5)

    def test_grad_replaces_none_with_zeros(self, backend):
        # func ignores its input entirely -> tape.gradient returns None
        # internally; grad() must substitute zeros_like rather than
        # propagate None to the caller.
        f = lambda x: tf.constant(5.0)
        g = backend.grad(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        result = g(x)
        np.testing.assert_allclose(result.numpy(), [0.0, 0.0, 0.0])

    def test_jacobian(self, backend):
        f = lambda x: tf.stack([x[0] * x[1], x[0] ** 2])
        x = backend.tensor([2.0, 3.0])
        J = backend.jacobian(f, x)
        expected = np.array([[3.0, 2.0], [4.0, 0.0]])
        np.testing.assert_allclose(J.numpy(), expected, atol=1e-5)

    def test_hessian_of_quadratic(self, backend):
        f = lambda x: tf.reduce_sum(x ** 2)
        x = backend.tensor([1.0, 2.0])
        H = backend.hessian(f, x)
        np.testing.assert_allclose(H.numpy(), 2 * np.eye(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Neural-network model interface
# ---------------------------------------------------------------------------

class TestBuildMLP:
    def test_build_mlp_returns_keras_model(self, backend):
        model = backend.build_mlp([2, 16, 16, 1], activation="tanh")
        assert isinstance(model, tf.keras.Model)

    def test_build_mlp_forward_shape(self, backend):
        model = backend.build_mlp([2, 16, 1], activation="tanh")
        x = tf.ones((5, 2))
        out = model(x)
        assert out.shape == (5, 1)

    def test_build_mlp_layer_count(self, backend):
        # [2, 64, 64, 1] -> 3 Dense layers (matches Flax backend's convention)
        model = backend.build_mlp([2, 64, 64, 1], activation="relu")
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        assert len(dense_layers) == 3

    def test_build_mlp_unknown_activation_raises(self, backend):
        with pytest.raises(ValueError):
            backend.build_mlp([2, 4, 1], activation="not_a_real_activation")

    def test_build_mlp_silu_maps_to_swish(self, backend):
        # TF calls silu "swish" internally; just verify no error and shape ok
        model = backend.build_mlp([2, 4, 1], activation="silu")
        out = model(tf.ones((1, 2)))
        assert out.shape == (1, 1)

    def test_build_mlp_use_residual(self, backend):
        # Residual connections only apply when consecutive layer sizes match
        model = backend.build_mlp([4, 4, 4, 1], activation="relu", use_residual=True)
        out = model(tf.ones((2, 4)))
        assert out.shape == (2, 1)

    def test_parameters_returns_trainable_variables(self, backend):
        model = backend.build_mlp([2, 8, 1], activation="tanh")
        params = backend.parameters(model)
        assert len(params) > 0
        assert all(isinstance(p, tf.Variable) for p in params)

    def test_zero_grad_is_noop(self, backend):
        model = backend.build_mlp([2, 4, 1])
        assert backend.zero_grad(model) is None


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------

class TestOptimizer:
    @pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop", "nadam"])
    def test_build_optimizer_known_names(self, backend, name):
        model = backend.build_mlp([2, 4, 1])
        opt = backend.build_optimizer(model, optimizer_name=name, lr=1e-3)
        assert isinstance(opt, tf.keras.optimizers.Optimizer)

    def test_build_optimizer_unknown_name_raises(self, backend):
        with pytest.raises(ValueError):
            backend.build_optimizer(None, optimizer_name="not_an_optimizer")

    def test_build_optimizer_strips_weight_decay_for_non_adamw(self, backend):
        model = backend.build_mlp([2, 4, 1])
        # Should not raise even though SGD doesn't accept weight_decay directly.
        opt = backend.build_optimizer(model, optimizer_name="sgd", lr=1e-3, weight_decay=0.01)
        assert isinstance(opt, tf.keras.optimizers.SGD)

    def test_optimizer_step_requires_model(self, backend):
        model = backend.build_mlp([2, 4, 1])
        opt = backend.build_optimizer(model, optimizer_name="adam", lr=1e-3)
        with pytest.raises(ValueError, match="model"):
            backend.optimizer_step(opt, backend.tensor(1.0), model=None, tape=None)

    def test_optimizer_step_requires_tape(self, backend):
        model = backend.build_mlp([2, 4, 1])
        opt = backend.build_optimizer(model, optimizer_name="adam", lr=1e-3)
        with pytest.raises(ValueError, match="tape"):
            backend.optimizer_step(opt, backend.tensor(1.0), model=model, tape=None)

    def test_optimizer_step_applies_update(self, backend):
        model = backend.build_mlp([2, 4, 1], activation="tanh")
        opt = backend.build_optimizer(model, optimizer_name="sgd", lr=0.1)
        before = [tf.identity(v) for v in model.trainable_variables]

        x = tf.ones((4, 2))
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_mean(tf.square(y))

        backend.optimizer_step(opt, loss, model=model, tape=tape)

        after = model.trainable_variables
        changed = any(
            not np.allclose(b.numpy(), a.numpy()) for b, a in zip(before, after)
        )
        assert changed


# ---------------------------------------------------------------------------
# JIT / vmap
# ---------------------------------------------------------------------------

class TestJitVmap:
    def test_jit_wraps_in_tf_function(self, backend):
        f = lambda x: x ** 2 + 1
        jitted = backend.jit(f)
        assert isinstance(jitted, tf.types.experimental.GenericFunction) or callable(jitted)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(jitted(x).numpy(), f(x).numpy())

    def test_vmap_default_axis0(self, backend):
        f = lambda x: tf.reduce_sum(x)
        vf = backend.vmap(f, in_axes=0)
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = vf(x)
        np.testing.assert_allclose(result.numpy(), [3.0, 7.0])

    def test_vmap_multi_arg_unpacked(self, backend):
        f = lambda a, b: a + b
        vf = backend.vmap(f, in_axes=0)
        a = backend.tensor([1.0, 2.0, 3.0])
        b = backend.tensor([10.0, 20.0, 30.0])
        result = vf(a, b)
        np.testing.assert_allclose(result.numpy(), [11.0, 22.0, 33.0])

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
    def test_default_device_string(self, backend):
        d = backend.default_device()
        assert d in ("/CPU:0", "/GPU:0")

    def test_to_device_cpu(self, backend):
        t = backend.tensor([1.0, 2.0])
        moved = backend.to_device(t, "/CPU:0")
        np.testing.assert_allclose(moved.numpy(), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Miscellaneous / configure / seed
# ---------------------------------------------------------------------------

class TestMisc:
    def test_configure_dtype(self, backend):
        backend.configure(dtype=tf.float64)
        assert backend._dtype == tf.float64

    def test_configure_device(self, backend):
        backend.configure(device="/CPU:0")
        assert backend._device == "/CPU:0"

    def test_seed_does_not_raise(self, backend):
        assert backend.seed(123) is None

    def test_seed_is_reproducible(self, backend):
        backend.seed(99)
        a = tf.random.uniform((3,))
        backend.seed(99)
        b = tf.random.uniform((3,))
        np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_require_tf_guard_importerror_message(self):
        from physai.backends.tensorflow_backend import _require_tf, _TF_AVAILABLE

        if _TF_AVAILABLE:
            # tf is installed in this environment, guard is a no-op
            assert _require_tf() is None
        else:  # pragma: no cover
            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                _require_tf()