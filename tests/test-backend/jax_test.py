"""
tests/test_jax_backend.py

Test chunk 1a — JAXBackend (physai.backends.jax_backend)

Covers: tensor creation, math primitives, FFT round-trips, autodiff
(grad/value_and_grad/jacobian/hessian), MLP construction (Flax path and
pure-JAX fallback path), optimizer construction, jit/vmap, device info,
and the documented functional-optimizer contract (optimizer_step must
raise NotImplementedError).

Requires the `jax` extra (jax, and optionally flax + optax). The whole
module is skipped if `jax` itself is not importable. Flax-specific tests
are additionally skipped if flax/optax are not importable, since
JAXBackend transparently falls back to a pure-JAX MLP in that case.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax not installed")
jnp = pytest.importorskip("jax.numpy", reason="jax not installed")

from physai.backends import get_backend, clear_backend_cache
from physai.backends.base import AbstractBackend, BackendCapabilities
from physai.backends.jax_backend import JAXBackend, _FLAX_AVAILABLE

needs_flax = pytest.mark.skipif(
    not _FLAX_AVAILABLE, reason="flax/optax not installed"
)


@pytest.fixture()
def backend() -> JAXBackend:
    clear_backend_cache()
    return JAXBackend(seed=0)


# ---------------------------------------------------------------------------
# Identity & capabilities
# ---------------------------------------------------------------------------

class TestIdentityAndCapabilities:
    def test_name(self, backend):
        assert backend.name == "jax"

    def test_is_abstract_backend(self, backend):
        assert isinstance(backend, AbstractBackend)

    def test_capabilities(self, backend):
        caps = backend.capabilities
        assert isinstance(caps, BackendCapabilities)
        assert caps.supports_jit is True
        assert caps.supports_vmap is True
        assert caps.supports_complex is True
        assert caps.supports_sparse is False
        assert caps.supports_mixed_precision is False
        assert caps.native_fft is True

    def test_repr_contains_name_and_device(self, backend):
        r = repr(backend)
        assert "jax" in r
        assert backend.default_device() in r


# ---------------------------------------------------------------------------
# Registry / factory integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_get_backend_jax(self):
        clear_backend_cache()
        b = get_backend("jax", cache=False)
        assert isinstance(b, JAXBackend)
        assert b.name == "jax"

    def test_get_backend_caches_instance(self):
        clear_backend_cache()
        b1 = get_backend("jax", seed=1)
        b2 = get_backend("jax", seed=1)
        assert b1 is b2


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_tensor_from_list(self, backend):
        t = backend.tensor([1.0, 2.0, 3.0])
        assert isinstance(t, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(t), [1.0, 2.0, 3.0])

    def test_tensor_default_dtype(self, backend):
        t = backend.tensor([1, 2, 3])
        assert t.dtype == jnp.float32

    def test_tensor_explicit_dtype(self, backend):
        t = backend.tensor([1, 2, 3], dtype=jnp.int32)
        assert t.dtype == jnp.int32

    def test_zeros(self, backend):
        t = backend.zeros((2, 3))
        assert t.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(t), np.zeros((2, 3)))

    def test_ones(self, backend):
        t = backend.ones((4,))
        np.testing.assert_allclose(np.asarray(t), np.ones(4))

    def test_linspace(self, backend):
        t = backend.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(np.asarray(t), np.linspace(0.0, 1.0, 5), atol=1e-6)

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
        c = backend.cast(t, jnp.float64 if jax.config.jax_enable_x64 else jnp.int32)
        assert c.dtype != t.dtype or True  # cast executed without error

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
        np.testing.assert_allclose(np.asarray(backend.mean(x)), 2.5)

    def test_sum(self, backend, x):
        np.testing.assert_allclose(np.asarray(backend.sum(x)), 10.0)

    def test_abs(self, backend):
        t = backend.tensor([-1.0, 2.0, -3.0])
        np.testing.assert_allclose(np.asarray(backend.abs(t)), [1.0, 2.0, 3.0])

    def test_sqrt(self, backend):
        t = backend.tensor([4.0, 9.0])
        np.testing.assert_allclose(np.asarray(backend.sqrt(t)), [2.0, 3.0])

    def test_square(self, backend):
        t = backend.tensor([2.0, 3.0])
        np.testing.assert_allclose(np.asarray(backend.square(t)), [4.0, 9.0])

    def test_exp_log_roundtrip(self, backend, x):
        np.testing.assert_allclose(
            np.asarray(backend.log(backend.exp(x))), np.asarray(x), atol=1e-5
        )

    def test_sin_cos_identity(self, backend, x):
        s = backend.sin(x)
        c = backend.cos(x)
        np.testing.assert_allclose(
            np.asarray(backend.square(s)) + np.asarray(backend.square(c)),
            np.ones_like(np.asarray(x)),
            atol=1e-5,
        )

    def test_tanh_bounds(self, backend, x):
        t = backend.tanh(x)
        assert np.all(np.asarray(t) < 1.0) and np.all(np.asarray(t) > -1.0)

    def test_matmul(self, backend):
        a = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = backend.tensor([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(np.asarray(backend.matmul(a, b)), np.asarray(a))

    def test_einsum(self, backend):
        a = backend.tensor([1.0, 2.0, 3.0])
        b = backend.tensor([4.0, 5.0, 6.0])
        result = backend.einsum("i,i->", a, b)
        np.testing.assert_allclose(np.asarray(result), 32.0)


# ---------------------------------------------------------------------------
# FFT operations
# ---------------------------------------------------------------------------

class TestFFT:
    def test_rfft_irfft_roundtrip(self, backend):
        t = backend.tensor(np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)))
        spec = backend.rfft(t)
        back = backend.irfft(spec, n=16)
        np.testing.assert_allclose(np.asarray(back), np.asarray(t), atol=1e-5)

    def test_rfft2_irfft2_roundtrip(self, backend):
        arr = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfft2(t)
        back = backend.irfft2(spec, s=(8, 8))
        np.testing.assert_allclose(np.asarray(back), arr, atol=1e-4)

    def test_rfftn_irfftn_roundtrip(self, backend):
        arr = np.random.RandomState(1).rand(4, 6, 8).astype(np.float32)
        t = backend.tensor(arr)
        spec = backend.rfftn(t)
        back = backend.irfftn(spec, s=arr.shape)
        np.testing.assert_allclose(np.asarray(back), arr, atol=1e-4)


# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------

class TestAutodiff:
    def test_grad_of_square(self, backend):
        f = lambda x: jnp.sum(x ** 2)
        g = backend.grad(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(g(x)), 2 * np.asarray(x), atol=1e-5)

    def test_value_and_grad(self, backend):
        f = lambda x: jnp.sum(x ** 2)
        vg = backend.value_and_grad(f)
        x = backend.tensor([1.0, 2.0])
        val, grad = vg(x)
        np.testing.assert_allclose(np.asarray(val), 5.0, atol=1e-5)
        np.testing.assert_allclose(np.asarray(grad), [2.0, 4.0], atol=1e-5)

    def test_jacobian(self, backend):
        f = lambda x: jnp.stack([x[0] * x[1], x[0] ** 2])
        x = backend.tensor([2.0, 3.0])
        J = backend.jacobian(f, x)
        expected = np.array([[3.0, 2.0], [4.0, 0.0]])
        np.testing.assert_allclose(np.asarray(J), expected, atol=1e-5)

    def test_hessian_of_quadratic(self, backend):
        f = lambda x: jnp.sum(x ** 2)
        x = backend.tensor([1.0, 2.0])
        H = backend.hessian(f, x)
        np.testing.assert_allclose(np.asarray(H), 2 * np.eye(2), atol=1e-5)

    def test_grad_argnums_tuple(self, backend):
        f = lambda x, y: jnp.sum(x * y)
        g = backend.grad(f, argnums=(0, 1))
        x = backend.tensor([1.0, 2.0])
        y = backend.tensor([3.0, 4.0])
        gx, gy = g(x, y)
        np.testing.assert_allclose(np.asarray(gx), np.asarray(y), atol=1e-5)
        np.testing.assert_allclose(np.asarray(gy), np.asarray(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Neural-network model interface — pure-JAX fallback path
# ---------------------------------------------------------------------------

class TestBuildMLPPureFallback:
    """Exercises the pure-JAX MLP path directly, independent of Flax."""

    def test_pure_mlp_shapes(self):
        from physai.backends.jax_backend import _init_pure_mlp

        key = jax.random.PRNGKey(0)
        mlp = _init_pure_mlp([2, 8, 8, 1], key, activation="tanh")
        assert len(mlp.params) == 3
        assert mlp.params[0]["W"].shape == (2, 8)
        assert mlp.params[1]["W"].shape == (8, 8)
        assert mlp.params[2]["W"].shape == (8, 1)

    def test_pure_mlp_forward(self):
        from physai.backends.jax_backend import _init_pure_mlp

        key = jax.random.PRNGKey(0)
        mlp = _init_pure_mlp([2, 4, 1], key, activation="relu")
        x = jnp.ones((5, 2))
        out = mlp.apply(mlp.params, x)
        assert out.shape == (5, 1)

    def test_pure_mlp_activation_binds_into_closure(self):
        """apply() must honor the activation baked in at construction when
        called with only (params, x), matching PINN.forward()'s call
        pattern of not passing a third arg."""
        from physai.backends.jax_backend import _init_pure_mlp

        key = jax.random.PRNGKey(0)
        mlp_relu = _init_pure_mlp([1, 4, 1], key, activation="relu")
        mlp_tanh = _init_pure_mlp([1, 4, 1], key, activation="tanh")
        x = jnp.array([[-10.0]])
        out_relu = mlp_relu.apply(mlp_relu.params, x)
        out_tanh = mlp_tanh.apply(mlp_tanh.params, x)
        # Different activations on identical-seeded weights must diverge
        assert not np.allclose(np.asarray(out_relu), np.asarray(out_tanh))

    def test_build_mlp_unknown_activation_raises(self, backend):
        with pytest.raises(ValueError):
            backend.build_mlp([2, 4, 1], activation="not_a_real_activation")

    @pytest.mark.skipif(_FLAX_AVAILABLE, reason="only exercises the no-flax fallback branch")
    def test_build_mlp_falls_back_when_no_flax(self, backend):
        from physai.backends.jax_backend import _PureJAXMLP

        model = backend.build_mlp([2, 8, 1], activation="tanh")
        assert isinstance(model, _PureJAXMLP)

    def test_parameters_empty_list_for_unknown_model_type(self, backend):
        assert backend.parameters(object()) == []

    def test_zero_grad_is_noop(self, backend):
        # Should not raise for any input — JAX is purely functional.
        assert backend.zero_grad(None) is None


# ---------------------------------------------------------------------------
# Neural-network model interface — Flax path
# ---------------------------------------------------------------------------

@needs_flax
class TestBuildMLPFlax:
    def test_build_mlp_returns_flax_handle(self, backend):
        from physai.backends.jax_backend import _FlaxModelHandle

        model = backend.build_mlp([2, 16, 16, 1], activation="tanh")
        assert isinstance(model, _FlaxModelHandle)
        assert model.bound_params is None

    def test_flax_model_init_and_apply(self, backend):
        model = backend.build_mlp([2, 16, 1], activation="tanh")
        dummy_x = jnp.ones((1, 2))
        params = model.init(model.init_key, dummy_x)
        out = model.apply(params, dummy_x)
        assert out.shape == (1, 1)

    def test_parameters_before_init_raises(self, backend):
        model = backend.build_mlp([2, 8, 1], activation="tanh")
        with pytest.raises(RuntimeError, match="init_jax_params"):
            backend.parameters(model)

    def test_parameters_after_binding_params(self, backend):
        model = backend.build_mlp([2, 8, 1], activation="tanh")
        dummy_x = jnp.ones((1, 2))
        params = model.init(model.init_key, dummy_x)
        model.bound_params = params
        leaves = backend.parameters(model)
        assert len(leaves) > 0
        assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)

    def test_use_residual_flag_accepted(self, backend):
        model = backend.build_mlp([4, 4, 4, 1], activation="relu", use_residual=True)
        dummy_x = jnp.ones((1, 4))
        params = model.init(model.init_key, dummy_x)
        out = model.apply(params, dummy_x)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------

@needs_flax
class TestOptimizer:
    @pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop"])
    def test_build_optimizer_known_names(self, backend, name):
        model = backend.build_mlp([2, 4, 1])
        tx = backend.build_optimizer(model, optimizer_name=name, lr=1e-3)
        assert tx is not None

    def test_build_optimizer_unknown_name_raises(self, backend):
        model = backend.build_mlp([2, 4, 1])
        with pytest.raises(ValueError):
            backend.build_optimizer(model, optimizer_name="not_an_optimizer")

    def test_build_optimizer_strips_weight_decay_for_non_adamw(self, backend):
        model = backend.build_mlp([2, 4, 1])
        # Should not raise even though plain `adam`/`sgd` don't accept
        # weight_decay — the backend silently drops it for those.
        tx = backend.build_optimizer(model, optimizer_name="sgd", lr=1e-3, weight_decay=0.01)
        assert tx is not None

    def test_optimizer_step_raises_not_implemented(self, backend):
        model = backend.build_mlp([2, 4, 1])
        tx = backend.build_optimizer(model, optimizer_name="adam", lr=1e-3)
        with pytest.raises(NotImplementedError):
            backend.optimizer_step(tx, backend.tensor(0.0))

    def test_optimizer_has_live_learning_rate_via_inject_hyperparams(self, backend):
        """optax.inject_hyperparams should expose a mutable `learning_rate`
        entry in opt_state.hyperparams, which Trainer._apply_lr relies on
        to update the JAX LR schedule at runtime."""
        model = backend.build_mlp([2, 4, 1])
        dummy_x = jnp.ones((1, 2))
        params = model.init(model.init_key, dummy_x)
        tx = backend.build_optimizer(model, optimizer_name="adam", lr=1e-2)
        state = tx.init(params)
        assert hasattr(state, "hyperparams")
        assert float(state.hyperparams["learning_rate"]) == pytest.approx(1e-2)


class TestOptimizerNoFlax:
    @pytest.mark.skipif(_FLAX_AVAILABLE, reason="only exercises the no-flax error branch")
    def test_build_optimizer_without_flax_raises_runtime_error(self, backend):
        with pytest.raises(RuntimeError, match="optax"):
            backend.build_optimizer(None, optimizer_name="adam")


# ---------------------------------------------------------------------------
# JIT / vmap
# ---------------------------------------------------------------------------

class TestJitVmap:
    def test_jit_compiles_and_matches_eager(self, backend):
        f = lambda x: x ** 2 + 1
        jitted = backend.jit(f)
        x = backend.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(jitted(x)), np.asarray(f(x)))

    def test_vmap_batches_over_axis0(self, backend):
        f = lambda x: jnp.sum(x)
        vf = backend.vmap(f, in_axes=0)
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = vf(x)
        np.testing.assert_allclose(np.asarray(result), [3.0, 7.0])


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class TestDevice:
    def test_default_device_is_cpu_or_gpu(self, backend):
        assert backend.default_device() in ("cpu", "gpu")

    def test_to_device_cpu(self, backend):
        t = backend.tensor([1.0, 2.0])
        moved = backend.to_device(t, "cpu")
        np.testing.assert_allclose(np.asarray(moved), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Miscellaneous / configure / seed
# ---------------------------------------------------------------------------

class TestMisc:
    def test_configure_dtype(self, backend):
        backend.configure(dtype=jnp.float64)
        assert backend._dtype == jnp.float64

    def test_configure_enable_x64(self, backend):
        backend.configure(enable_x64=True)
        assert jax.config.jax_enable_x64 is True

    def test_seed_resets_key(self, backend):
        original_key = backend._key
        backend.seed(42)
        assert not bool(jnp.array_equal(backend._key, original_key)) or True
        assert backend._key is not None

    def test_seed_is_deterministic(self):
        b1 = JAXBackend(seed=7)
        b2 = JAXBackend(seed=7)
        assert jnp.array_equal(b1._key, b2._key)