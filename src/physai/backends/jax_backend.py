"""
physai/backends/jax_backend.py

JAX implementation of the AbstractBackend interface.

Design notes
------------
* JAX arrays are immutable; stateful operations (parameter updates) are
  handled through ``optax`` (or ``jax.example_libraries.optimizers`` as
  fallback).
* ``jax.grad`` / ``jax.value_and_grad`` naturally support higher-order
  derivatives when ``has_aux=False`` and when composed recursively.
* ``vmap`` and ``jit`` are native JAX primitives — this backend exposes
  them with a consistent signature.
* We use ``jax.numpy`` for all array math and ``jax.scipy.fft`` for
  spectral operations (available in JAX ≥ 0.3.0).
* Neural networks are built with ``flax.linen`` for production quality.
  If Flax is unavailable, a pure-JAX fallback MLP is used.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.fft as jax_fft
import numpy as np

from .base import AbstractBackend, BackendCapabilities, DType, Shape, Tensor

# ---------------------------------------------------------------------------
# Optional Flax dependency
# ---------------------------------------------------------------------------
try:
    import flax.linen as nn # type: ignore
    import optax # type: ignore
    _FLAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FLAX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------
_ACTIVATIONS: Dict[str, Callable] = {
    "tanh":    jnp.tanh,
    "relu":    jax.nn.relu,
    "gelu":    jax.nn.gelu,
    "silu":    jax.nn.silu,
    "sigmoid": jax.nn.sigmoid,
    "elu":     jax.nn.elu,
    "softplus": jax.nn.softplus,
    # Snake (Ziyin et al. 2020): x + sin(x)^2, fixed frequency a=1 (matches
    # torch_backend.Snake's default, non-learnable) -- no extra state
    # needed, so it's a plain stateless function like the others above.
    "snake":   lambda x: x + jnp.sin(x) ** 2,
}

# ---------------------------------------------------------------------------
# Stateful / learnable activations (L-LAAF, Pade)
# ---------------------------------------------------------------------------
# These carry their own trainable parameters, so -- unlike the plain
# functions in _ACTIVATIONS above -- they need an (init, apply) pair:
# `init()` returns the initial parameter pytree for one layer's activation
# instance, and `apply(x, params)` evaluates it. Kept out of _ACTIVATIONS
# because that dict's contract (a bare Callable[[Tensor], Tensor]) has no
# way to carry per-layer learnable state.

def _make_llaaf_apply(n: float = 10.0) -> Callable:
    """
    L-LAAF (Jagtap & Karniadakis 2020): a*tanh(n*a*x), `a` learnable per
    layer, `n` a fixed scaling factor closed over here (kept out of the
    params pytree so optax never tries to update it).
    """
    def _apply(x: jnp.ndarray, p: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        a = p["a"]
        # No outer 'a' coefficient — see torch_backend.py's
        # AdaptiveTanh.forward for the L-LAAF reasoning.
        return jnp.tanh(n * a * x)
    return _apply


def _pade_apply(x: jnp.ndarray, p: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Safe Pade Activation Unit (Molina et al. 2019): P(x)/Q(x), Q's
    coefficients under abs(...) so the denominator never vanishes."""
    p0, p1, p2, p3 = p["p"]
    q1, q2 = p["q"]
    numerator = p0 + p1 * x + p2 * x ** 2 + p3 * x ** 3
    denominator = 1.0 + jnp.abs(q1 * x) + jnp.abs(q2 * x ** 2)
    return numerator / denominator


_STATEFUL_ACTIVATIONS: Dict[str, Dict[str, Callable]] = {
    "llaaf": {
        # a*n = 1 at init -> starts identical to plain tanh(x).
        "init":  lambda: {"a": jnp.array(1.0 / 10.0)},
        "apply": _make_llaaf_apply(10.0),
    },
    "pade": {
        "init":  lambda: {
            "p": jnp.array([0.0, 1.0, 0.5, 0.0]),
            "q": jnp.array([0.5, 0.0]),
        },
        "apply": _pade_apply,
    },
}


# ---------------------------------------------------------------------------
# Flax MLP (preferred path)
# ---------------------------------------------------------------------------
if _FLAX_AVAILABLE:
    class _FlaxMLP(nn.Module):
        layer_sizes:  Tuple[int, ...]
        activation:   str = "tanh"
        use_residual: bool = False

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            stateful = _STATEFUL_ACTIVATIONS.get(self.activation)
            act = None if stateful is not None else _ACTIVATIONS[self.activation]
            # layer_sizes[0] is the input dimension, not a Dense layer to
            # build. Iterate over the *output* sizes (layer_sizes[1:]) so
            # that len(layer_sizes) - 1 Dense layers are created, matching
            # what the caller asked for (e.g. [2, 64, 64, 1] -> 3 layers).
            n_layers = len(self.layer_sizes) - 1
            for i, size in enumerate(self.layer_sizes[1:]):
                is_last = i == n_layers - 1
                h = nn.Dense(size)(x)
                if not is_last:
                    if stateful is not None:
                        # Each layer gets its own learnable activation
                        # parameters via a uniquely-named Flax param
                        # (self.param dedupes/creates fresh state per
                        # distinct name, giving per-layer independence
                        # the same way torch_backend's fresh-instance-
                        # per-layer _MLP does).
                        init_vals = stateful["init"]()
                        act_params = {
                            k: self.param(f"act_{i}_{k}", lambda key, v=v: v)
                            for k, v in init_vals.items()
                        }
                        h = stateful["apply"](h, act_params)
                    else:
                        h = act(h)
                    if self.use_residual and x.shape[-1] == size:
                        x = x + h
                    else:
                        x = h
                else:
                    x = h
            return x


# ---------------------------------------------------------------------------
# Handle for Flax models (Flax Modules are frozen dataclasses)
# ---------------------------------------------------------------------------

if _FLAX_AVAILABLE:
    class _FlaxModelHandle:
        """Mutable wrapper carrying state for a frozen Flax Module.

        ``flax.linen.Module`` instances are frozen dataclasses, so
        attaching arbitrary state (an init PRNG key, bound parameters)
        directly to the module raises
        ``flax.errors.SetAttributeFrozenModuleError``. This handle holds
        that state alongside the module instead, while forwarding
        ``init``/``apply`` so callers can keep treating it like a model.
        """

        __slots__ = ("module", "init_key", "bound_params")

        def __init__(self, module: "nn.Module", init_key: jnp.ndarray) -> None:
            self.module = module
            self.init_key = init_key
            self.bound_params: Optional[Any] = None

        def init(self, key: jnp.ndarray, x: jnp.ndarray) -> Any:
            return self.module.init(key, x)

        def apply(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
            return self.module.apply(params, x)


# ---------------------------------------------------------------------------
# Pure-JAX fallback MLP (params dict + apply fn)
# ---------------------------------------------------------------------------

class _PureJAXMLP(NamedTuple):
    """Lightweight MLP container for environments without Flax."""
    params: Any
    apply:  Callable


def _init_pure_mlp(
    layer_sizes: Sequence[int],
    key: jnp.ndarray,
    activation: str = "tanh",
) -> _PureJAXMLP:
    params = []
    for i in range(len(layer_sizes) - 1):
        k1, k2, key = jax.random.split(key, 3)
        W = jax.random.normal(k1, (layer_sizes[i], layer_sizes[i + 1])) * 0.1
        b = jnp.zeros((layer_sizes[i + 1],))
        layer_params: Dict[str, Any] = {"W": W, "b": b}
        is_last = i == len(layer_sizes) - 2
        stateful = _STATEFUL_ACTIVATIONS.get(activation)
        if not is_last and stateful is not None:
            # Each hidden layer gets its own independent activation-
            # parameter pytree (mirrors torch_backend's fresh-instance-
            # per-layer _MLP and the Flax path's uniquely-named
            # self.param per layer above).
            layer_params["act"] = stateful["init"]()
        params.append(layer_params)

    # Bind the caller's chosen activation into the returned `apply`
    # closure via a default arg, so callers that invoke
    # `model.apply(params, x)` without a third argument (as PINN.forward()
    # does) get the requested activation instead of silently falling back
    # to "tanh".
    def apply(p: List[Dict], x: jnp.ndarray, activation: str = activation) -> jnp.ndarray:
        stateful = _STATEFUL_ACTIVATIONS.get(activation)
        act = None if stateful is not None else _ACTIVATIONS[activation]
        for i, layer in enumerate(p):
            x = x @ layer["W"] + layer["b"]
            if i < len(p) - 1:
                x = stateful["apply"](x, layer["act"]) if stateful is not None else act(x)
        return x

    return _PureJAXMLP(params=params, apply=apply)


# ---------------------------------------------------------------------------
# Optimiser state wrapper (optax or fallback)
# ---------------------------------------------------------------------------

class _OptaxOptimizer:
    """Thin wrapper that keeps optax state together with the update function."""

    def __init__(self, tx: Any, params: Any) -> None:
        self.tx     = tx
        self.state  = tx.init(params)
        self._params_ref: List = []  # mutable reference bucket

    def step(self, grads: Any, params: Any) -> Any:
        updates, self.state = self.tx.update(grads, self.state, params)
        return optax.apply_updates(params, updates)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class JAXBackend(AbstractBackend):
    """JAX execution backend for PhysAI."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Any = jnp.float32,
        *,
        enable_x64: bool = False,
        seed: int = 0,
    ) -> None:
        if enable_x64:
            jax.config.update("jax_enable_x64", True)
        self._dtype  = dtype
        try:
            # Try querying the GPU devices directly
            self.device = "gpu" if len(jax.devices("gpu")) > 0 else "cpu"
        except (RuntimeError, ValueError):
            # If JAX throws an error because no GPU platform exists, use CPU
            self.device = "cpu"


        self._key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "jax"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jit             = True,
            supports_vmap            = True,
            supports_complex         = True,
            supports_sparse          = False,
            supports_mixed_precision = False,
            native_fft               = True,
        )

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def tensor(self, data: Any, dtype=None, device=None) -> jnp.ndarray:
        return jnp.array(data, dtype=dtype or self._dtype)

    def zeros(self, shape: Shape, dtype=None) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=dtype or self._dtype)

    def ones(self, shape: Shape, dtype=None) -> jnp.ndarray:
        return jnp.ones(shape, dtype=dtype or self._dtype)

    def linspace(self, start: float, stop: float, num: int, dtype=None) -> jnp.ndarray:
        return jnp.linspace(start, stop, num, dtype=dtype or self._dtype)

    def meshgrid(self, *tensors: jnp.ndarray, indexing: str = "ij") -> List[jnp.ndarray]:
        return list(jnp.meshgrid(*tensors, indexing=indexing))

    def stack(self, tensors: Sequence[jnp.ndarray], axis: int = 0) -> jnp.ndarray:
        return jnp.stack(list(tensors), axis=axis)

    def concatenate(self, tensors: Sequence[jnp.ndarray], axis: int = 0) -> jnp.ndarray:
        return jnp.concatenate(list(tensors), axis=axis)

    def reshape(self, tensor: jnp.ndarray, shape: Shape) -> jnp.ndarray:
        return jnp.reshape(tensor, shape)

    def cast(self, tensor: jnp.ndarray, dtype: Any) -> jnp.ndarray:
        return tensor.astype(dtype)

    def to_numpy(self, tensor: jnp.ndarray) -> np.ndarray:
        return np.asarray(tensor)

    # ------------------------------------------------------------------
    # Mathematical primitives
    # ------------------------------------------------------------------

    def mean(self, tensor: jnp.ndarray, axis=None) -> jnp.ndarray:
        return jnp.mean(tensor, axis=axis)

    def sum(self, tensor: jnp.ndarray, axis=None) -> jnp.ndarray:
        return jnp.sum(tensor, axis=axis)

    def abs(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(tensor)

    def sqrt(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(tensor)

    def square(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.square(tensor)

    def exp(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(tensor)

    def log(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(tensor)

    def sin(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(tensor)

    def cos(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(tensor)

    def atan(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctan(tensor)

    def tanh(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(tensor)

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def einsum(self, equation: str, *operands: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum(equation, *operands)

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    def rfft(self, tensor: jnp.ndarray, n=None, axis: int = -1) -> jnp.ndarray:
        return jnp.fft.rfft(tensor, n=n, axis=axis)

    def irfft(self, tensor: jnp.ndarray, n=None, axis: int = -1) -> jnp.ndarray:
        return jnp.fft.irfft(tensor, n=n, axis=axis)

    def rfft2(self, tensor: jnp.ndarray, s=None) -> jnp.ndarray:
        return jnp.fft.rfft2(tensor, s=s)

    def irfft2(self, tensor: jnp.ndarray, s=None) -> jnp.ndarray:
        return jnp.fft.irfft2(tensor, s=s)

    def rfftn(self, tensor: jnp.ndarray, s=None, axes=None) -> jnp.ndarray:
        return jnp.fft.rfftn(tensor, s=s, axes=axes)

    def irfftn(self, tensor: jnp.ndarray, s=None, axes=None) -> jnp.ndarray:
        return jnp.fft.irfftn(tensor, s=s, axes=axes)

    # ------------------------------------------------------------------
    # Automatic differentiation
    # ------------------------------------------------------------------

    def jvp(
        self,
        func: Callable[..., jnp.ndarray],
        inputs: jnp.ndarray,
        tangents: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Native forward-mode AD via ``jax.jvp``."""
        return jax.jvp(func, (inputs,), (tangents,))

    def grad(
        self,
        func: Callable[..., jnp.ndarray],
        argnums: Union[int, Sequence[int]] = 0,
        *,
        mode: str = "reverse",
    ) -> Callable[..., jnp.ndarray]:
        """
        ``mode="reverse"`` (default) wraps ``jax.grad`` — naturally supports
        higher-order composition and is cheapest for the common PINN case
        (scalar/small-vector output, many collocation points).

        ``mode="forward"`` wraps ``jax.jacfwd`` — cheaper when ``func`` has
        many output components relative to input width (coupled/mixed
        residual systems).

        ``mode="taylor"`` uses ``jax.experimental.jet`` to pull second-order
        directional (Taylor coefficient) derivatives out of a single
        forward sweep — used for cheap Laplacian diagonals.
        """
        if mode == "reverse":
            return jax.grad(func, argnums=argnums)
        if mode == "forward":
            return jax.jacfwd(func, argnums=argnums)
        if mode == "taylor":
            from jax.experimental import jet

            def taylor_grad(*args: Any, **kwargs: Any) -> Any:
                argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)
                outs = []
                for idx in argnums_list:
                    x = args[idx]
                    n_dims = x.shape[-1]
                    cols = []
                    for d in range(n_dims):
                        tangent = jnp.zeros_like(x).at[..., d].set(1.0)

                        def f_single(xi):
                            call_args = list(args)
                            call_args[idx] = xi
                            return func(*call_args, **kwargs)

                        # jet gives Taylor coefficients [f, f', f''/2!, ...]
                        # along `tangent`; series=[[zeros]] seeds a pure
                        # second directional derivative in one forward pass.
                        _, (_, second) = jet.jet(
                            f_single, (x,), ((tangent, jnp.zeros_like(x)),)
                        )
                        cols.append(second[..., None] if second.ndim == x.ndim - 1 else second)
                    outs.append(jnp.concatenate(cols, axis=-1) if cols[0].ndim else jnp.stack(cols, axis=-1))
                return outs[0] if len(outs) == 1 else tuple(outs)

            return taylor_grad
        raise ValueError(f"Unknown differentiation mode: {mode!r}")

    def value_and_grad(
        self,
        func: Callable[..., jnp.ndarray],
        argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]:
        return jax.value_and_grad(func, argnums=argnums)

    def jacobian(
        self,
        func: Callable[..., jnp.ndarray],
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.jacobian(func)(inputs)

    def hessian(
        self,
        func: Callable[..., jnp.ndarray],
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.hessian(func)(inputs)

    # ------------------------------------------------------------------
    # Neural-network model interface
    # ------------------------------------------------------------------

    def build_mlp(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        *,
        use_residual: bool = False,
        dtype=None,
    ) -> Any:
        if activation not in _ACTIVATIONS and activation not in _STATEFUL_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from "
                f"{list(_ACTIVATIONS) + list(_STATEFUL_ACTIVATIONS)}."
            )
        if _FLAX_AVAILABLE:
            module = _FlaxMLP(
                layer_sizes  = tuple(layer_sizes),
                activation   = activation,
                use_residual = use_residual,
            )
            self._key, subkey = jax.random.split(self._key)
            # Lazily initialise — caller must call model.init(key, dummy_x).
            # We can't attach the init key directly to `module` (Flax
            # Modules are frozen dataclasses), so wrap it in a mutable
            # handle instead.
            return _FlaxModelHandle(module=module, init_key=subkey)
        else:
            self._key, subkey = jax.random.split(self._key)
            return _init_pure_mlp(list(layer_sizes), subkey, activation=activation)

    def parameters(self, model: Any) -> List[Any]:
        """Return the flattened parameter pytree leaves."""
        if _FLAX_AVAILABLE and isinstance(model, _FlaxModelHandle):
            # Flax modules are stateless; the bound params dict must be
            # attached to the handle by PINN.init_jax_params() (as
            # `bound_params`) before this can return anything real.
            params = model.bound_params
            if params is None:
                raise RuntimeError(
                    "This Flax model has no initialised parameters yet. "
                    "Call PINN.init_jax_params(dummy_input) before requesting "
                    "parameters(), named_parameters(), save(), or repr()."
                )
            return jax.tree_util.tree_leaves(params)
        if isinstance(model, _PureJAXMLP):
            return jax.tree_util.tree_leaves(model.params)
        return []

    def zero_grad(self, model: Any) -> None:
        pass  # JAX is functional — no accumulated grads to clear

    # ------------------------------------------------------------------
    # Optimiser helpers
    # ------------------------------------------------------------------

    def build_optimizer(
        self,
        model: Any,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> Any:
        if not _FLAX_AVAILABLE:
            raise RuntimeError("optax is required for JAX optimisers. Install flax + optax.")

        _OPT_MAP = {
            "adam":    optax.adam,
            "adamw":   optax.adamw,
            "sgd":     optax.sgd,
            "rmsprop": optax.rmsprop,
        }
        cls = _OPT_MAP.get(optimizer_name.lower())
        if cls is None:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose from {list(_OPT_MAP)}.")
        
        if optimizer_name.lower() not in {"adamw", "adamax", "lamb"}:
            kwargs.pop("weight_decay", None)  # Only adamw supports weight decay

        # Wrap with inject_hyperparams so `learning_rate` becomes a live,
        # mutable entry in opt_state rather than baked in at construction.
        # This is what lets Trainer._apply_lr update the JAX LR schedule
        # at runtime (see trainer.py), matching the torch/tensorflow paths
        # instead of silently no-op'ing on JAX.
        tx = optax.inject_hyperparams(cls)(learning_rate=lr, **kwargs)
        return tx  # caller stores (tx, opt_state) separately

    def optimizer_step(
        self,
        optimizer: Any,
        loss: jnp.ndarray,
        *,
        model: Optional[Any] = None,
        tape: Optional[Any] = None,
    ) -> None:
        """
        JAX optimisation is functional. This method is a no-op placeholder;
        the recommended pattern is:

            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        """
        raise NotImplementedError(
            "JAX optimizer_step is intentionally not implemented here because "
            "JAX optimization is purely functional. "
            "Use: updates, state = tx.update(grads, state, params) "
            "followed by optax.apply_updates(params, updates)."
        )

    # ------------------------------------------------------------------
    # JIT / vmap
    # ------------------------------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        return jax.jit(func, **kwargs)

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, Sequence[Optional[int]]] = 0,
        out_axes: int = 0,
    ) -> Callable:
        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def default_device(self) -> str:
        return self.device

    def to_device(self, tensor: jnp.ndarray, device: str) -> jnp.ndarray:
        target = jax.devices(device)[0]
        return jax.device_put(tensor, target)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        if "enable_x64" in kwargs:
            jax.config.update("jax_enable_x64", kwargs["enable_x64"])
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]

    def seed(self, value: int) -> None:
        self._key = jax.random.PRNGKey(value)