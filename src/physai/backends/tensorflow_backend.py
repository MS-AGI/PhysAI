# pyright: reportInvalidTypeForm=false

"""
physai/backends/tensorflow_backend.py

TensorFlow / Keras implementation of the AbstractBackend interface.

Design notes
------------
* All differentiation goes through ``tf.GradientTape``. To support
  higher-order derivatives (needed for PDE residuals such as ∂²u/∂x²),
  tapes are nested with ``persistent=True`` and the outer tape watches
  the inner gradient.
* We never call ``.numpy()`` inside a ``@tf.function`` — numpy conversion
  is only performed in ``to_numpy()``, which is necessarily eager.
* Models are built with ``tf.keras`` (the unified API available in TF ≥ 2.x).
* FFT operations delegate to ``tf.signal`` which wraps cuFFT/FFTW.
* The optimizer step signature accepts an optional ``tape`` and ``model``
  to support the standard TF training loop pattern.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Lazy import so the module is importable even when TF is not installed
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    tf = None  

from .base import AbstractBackend, BackendCapabilities, DType, Shape, Tensor

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _require_tf() -> None:  # pragma: no cover
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. "
            "Install it with: pip install tensorflow"
        )


# ---------------------------------------------------------------------------
# Activation & optimizer registries
# ---------------------------------------------------------------------------

_ACTIVATIONS_STR: Dict[str, str] = {
    "tanh":    "tanh",
    "relu":    "relu",
    "gelu":    "gelu",
    "silu":    "swish",   # TF calls it swish
    "sigmoid": "sigmoid",
    "elu":     "elu",
    "softplus": "softplus",
    "mish":    "mish",
}


class SnakeLayer(tf.keras.layers.Layer if _TF_AVAILABLE else object):
    """
    Snake activation (Ziyin et al. 2020): x + sin(x)^2. Fixed frequency
    a=1, matching torch_backend.Snake's non-learnable default -- no
    add_weight() call needed, unlike the two learnable layers below.
    """
    def call(self, x: "tf.Tensor") -> "tf.Tensor":
        return x + tf.sin(x) ** 2


class LLAAFLayer(tf.keras.layers.Layer if _TF_AVAILABLE else object):
    """
    Layer-wise Locally Adaptive Activation Function, L-LAAF
    (Jagtap & Karniadakis 2020): a*tanh(n*a*x), `a` a trainable scalar
    weight of this layer, `n` a fixed scaling factor.
    """
    def __init__(self, n: float = 10.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape) -> None:
        # a*n = 1 at init -> starts identical to plain tanh(x).
        self.a = self.add_weight(
            name="a", shape=(),
            initializer=tf.keras.initializers.Constant(1.0 / self.n),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: "tf.Tensor") -> "tf.Tensor":
        return tf.tanh(self.n * self.a * x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"n": self.n})
        return config


class PadeLayer(tf.keras.layers.Layer if _TF_AVAILABLE else object):
    """
    Safe Pade Activation Unit, PAU (Molina et al. 2019): learnable
    rational P(x)/Q(x), P degree 3 / Q degree 2, Q's coefficients under
    abs(...) so the denominator never vanishes.
    """
    def build(self, input_shape) -> None:
        self.p = self.add_weight(
            name="p", shape=(4,),
            initializer=tf.keras.initializers.Constant([0.0, 1.0, 0.5, 0.0]),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q", shape=(2,),
            initializer=tf.keras.initializers.Constant([0.5, 0.0]),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: "tf.Tensor") -> "tf.Tensor":
        p0, p1, p2, p3 = self.p[0], self.p[1], self.p[2], self.p[3]
        q1, q2 = self.q[0], self.q[1]
        numerator = p0 + p1 * x + p2 * x ** 2 + p3 * x ** 3
        denominator = 1.0 + tf.abs(q1 * x) + tf.abs(q2 * x ** 2)
        return numerator / denominator


# Custom-layer activations (stateless or learnable) that can't be
# expressed as a bare Keras built-in activation string like the ones in
# _ACTIVATIONS_STR above -- each entry is a *factory*, called fresh at
# each use site (see _build_keras_mlp) so learnable ones get independent
# weights per layer.
_CUSTOM_ACTIVATION_LAYERS: Dict[str, Callable[[], "tf.keras.layers.Layer"]] = {
    "snake": SnakeLayer,
    "llaaf": LLAAFLayer,
    "pade":  PadeLayer,
}


def _build_tf_optimizer(name: str, lr: float, **kwargs: Any) -> "tf.keras.optimizers.Optimizer":
    _require_tf()
    registry: Dict[str, Any] = {
        "adam":    tf.keras.optimizers.Adam,
        "adamw":   tf.keras.optimizers.AdamW,
        "sgd":     tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "nadam":   tf.keras.optimizers.Nadam,
    }
    cls = registry.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from {list(registry)}."
        )
    return cls(learning_rate=lr, **kwargs)


# ---------------------------------------------------------------------------
# Residual MLP (Keras functional API)
# ---------------------------------------------------------------------------

def _build_keras_mlp(
    layer_sizes: Sequence[int],
    activation: str,
    use_residual: bool,
    dtype: Any,
) -> "tf.keras.Model":
    _require_tf()
    activation_lower = activation.lower()
    act_name = _ACTIVATIONS_STR.get(activation_lower)
    custom_layer_cls = _CUSTOM_ACTIVATION_LAYERS.get(activation_lower)
    if act_name is None and custom_layer_cls is None:
        raise ValueError(
            f"Unknown activation '{activation}'. Choose from "
            f"{list(_ACTIVATIONS_STR) + list(_CUSTOM_ACTIVATION_LAYERS)}."
        )

    inputs = tf.keras.Input(shape=(layer_sizes[0],), dtype=dtype)
    x = inputs

    for i, units in enumerate(layer_sizes[1:]):
        is_last = i == len(layer_sizes) - 2
        h = tf.keras.layers.Dense(units, dtype=dtype)(x)
        if not is_last:
            if custom_layer_cls is not None:
                # Fresh instance per layer -- gives learnable activations
                # (L-LAAF/Pade) independent weights per layer, matching
                # the plain Activation(act_name) path below (which also
                # already constructs a new layer object each iteration).
                h = custom_layer_cls(dtype=dtype)(h)
            else:
                h = tf.keras.layers.Activation(act_name, dtype=dtype)(h)
            if use_residual and x.shape[-1] == units:
                x = tf.keras.layers.Add(dtype=dtype)([x, h])
            else:
                x = h
        else:
            x = h

    return tf.keras.Model(inputs=inputs, outputs=x)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class TensorFlowBackend(AbstractBackend):
    """TensorFlow execution backend for PhysAI."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Any = None,
        *,
        memory_growth: bool = True,
        mixed_precision: bool = False,
    ) -> None:
        _require_tf()

        # Configure GPU memory growth before any tensor ops
        if memory_growth:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # Already initialised

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self._dtype  = dtype or tf.float32
        self._device = device or (
            "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        )

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "tensorflow"

    @property
    def capabilities(self) -> BackendCapabilities:
        _require_tf()
        has_gpu = bool(tf.config.list_physical_devices("GPU"))
        return BackendCapabilities(
            supports_jit             = True,
            supports_vmap            = False,   # tf.vectorized_map is limited
            supports_complex         = True,
            supports_sparse          = True,
            supports_mixed_precision = has_gpu,
            native_fft               = True,
        )

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def tensor(self, data: Any, dtype=None, device=None) -> "tf.Tensor":
        dt = dtype or self._dtype
        with tf.device(device or self._device):
            return tf.cast(tf.constant(data), dtype=dt)

    def zeros(self, shape: Shape, dtype=None) -> "tf.Tensor":
        with tf.device(self._device):
            return tf.zeros(shape, dtype=dtype or self._dtype)

    def ones(self, shape: Shape, dtype=None) -> "tf.Tensor":
        with tf.device(self._device):
            return tf.ones(shape, dtype=dtype or self._dtype)

    def linspace(self, start: float, stop: float, num: int, dtype=None) -> "tf.Tensor":
        t = tf.linspace(float(start), float(stop), num)
        return tf.cast(t, dtype=dtype or self._dtype)

    def meshgrid(self, *tensors: "tf.Tensor", indexing: str = "ij") -> List["tf.Tensor"]:
        return list(tf.meshgrid(*tensors, indexing=indexing))

    def stack(self, tensors: Sequence["tf.Tensor"], axis: int = 0) -> "tf.Tensor":
        return tf.stack(list(tensors), axis=axis)

    def concatenate(self, tensors: Sequence["tf.Tensor"], axis: int = 0) -> "tf.Tensor":
        return tf.concat(list(tensors), axis=axis)

    def reshape(self, tensor: "tf.Tensor", shape: Shape) -> "tf.Tensor":
        return tf.reshape(tensor, shape)

    def cast(self, tensor: "tf.Tensor", dtype: Any) -> "tf.Tensor":
        return tf.cast(tensor, dtype=dtype)

    def to_numpy(self, tensor: "tf.Tensor") -> np.ndarray:
        return tensor.numpy()

    # ------------------------------------------------------------------
    # Mathematical primitives
    # ------------------------------------------------------------------

    def mean(self, tensor: "tf.Tensor", axis=None) -> "tf.Tensor": 
        return tf.reduce_mean(tensor, axis=axis)

    def sum(self, tensor: "tf.Tensor", axis=None) -> "tf.Tensor":
        return tf.reduce_sum(tensor, axis=axis)

    def abs(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.abs(tensor)

    def sqrt(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.sqrt(tensor)

    def square(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.square(tensor)

    def exp(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.exp(tensor)

    def log(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.math.log(tensor)

    def sin(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.sin(tensor)

    def cos(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.cos(tensor)

    def atan(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.atan(tensor)

    def tanh(self, tensor: "tf.Tensor") -> "tf.Tensor":
        return tf.tanh(tensor)

    def matmul(self, a: "tf.Tensor", b: "tf.Tensor") -> "tf.Tensor":
        return tf.matmul(a, b)

    def einsum(self, equation: str, *operands: "tf.Tensor") -> "tf.Tensor":
        return tf.einsum(equation, *operands)

    # ------------------------------------------------------------------
    # FFT  — delegates to tf.signal
    # ------------------------------------------------------------------

    def rfft(self, tensor: "tf.Tensor", n=None, axis: int = -1) -> "tf.Tensor":
        # tf.signal.rfft operates on the last axis; roll if needed, and
        # roll back afterwards so `axis` behaves symmetrically to irfft()
        # (rfftn's chaining of rfft + per-axis fft relies on this symmetry).
        needs_move = axis not in (-1, len(tensor.shape) - 1)
        if needs_move:
            tensor = tf.experimental.numpy.moveaxis(tensor, axis, -1)
        if n is not None:
            tensor = tensor[..., :n]
        result = tf.signal.rfft(tf.cast(tensor, tf.float32))
        result = tf.cast(result, tf.complex64)
        if needs_move:
            result = tf.experimental.numpy.moveaxis(result, -1, axis)
        return result

    def irfft(self, tensor: "tf.Tensor", n=None, axis: int = -1) -> "tf.Tensor":
        needs_move = axis not in (-1, len(tensor.shape) - 1)
        if needs_move:
            tensor = tf.experimental.numpy.moveaxis(tensor, axis, -1)
        result = tf.signal.irfft(tensor, fft_length=[n] if n else None)
        if needs_move:
            result = tf.experimental.numpy.moveaxis(result, -1, axis)
        return result

    def rfft2(self, tensor: "tf.Tensor", s=None) -> "tf.Tensor":
        inp = tf.cast(tensor, tf.float32)
        if s is not None:
            inp = inp[..., : s[0], : s[1]]
        return tf.signal.rfft2d(inp)

    def irfft2(self, tensor: "tf.Tensor", s=None) -> "tf.Tensor":
        return tf.signal.irfft2d(tensor, fft_length=list(s) if s else None)

    def rfftn(self, tensor: "tf.Tensor", s=None, axes=None) -> "tf.Tensor":
        """
        Real-input N-D FFT. TF has no native rfftn, so we do ONE real rfft
        on the first axis (real -> complex), then complex fft on every
        remaining axis. The previous implementation re-cast the already
        complex intermediate to float32 on each subsequent axis, silently
        dropping the imaginary part and corrupting the spectrum for any
        N-D (N >= 2) transform.
        """
        axes = list(axes) if axes is not None else list(range(len(tensor.shape)))
        first_ax, rest_axes = axes[0], axes[1:]

        inp = tf.cast(tensor, tf.float32)
        first_n = s[0] if s else None
        result = self.rfft(inp, n=first_n, axis=first_ax)  # real -> complex, only here

        for i, ax in enumerate(rest_axes, start=1):
            n = s[i] if s else None
            needs_move = ax not in (-1, len(result.shape) - 1)
            if needs_move:
                result = tf.experimental.numpy.moveaxis(result, ax, -1)
            if n is not None:
                result = result[..., :n]
            result = tf.signal.fft(result)
            if needs_move:
                result = tf.experimental.numpy.moveaxis(result, -1, ax)

        return result

    def irfftn(self, tensor: "tf.Tensor", s=None, axes=None) -> "tf.Tensor":
        """Inverse of rfftn: complex ifft on the trailing axes, real irfft on the first."""
        axes = list(axes) if axes is not None else list(range(len(tensor.shape)))
        first_ax, rest_axes = axes[0], axes[1:]

        result = tensor
        for i, ax in reversed(list(enumerate(rest_axes, start=1))):
            n = s[i] if s else None
            needs_move = ax not in (-1, len(result.shape) - 1)
            if needs_move:
                result = tf.experimental.numpy.moveaxis(result, ax, -1)
            result = tf.signal.ifft(result)
            if n is not None:
                result = result[..., :n]
            if needs_move:
                result = tf.experimental.numpy.moveaxis(result, -1, ax)

        first_n = s[0] if s else None
        return self.irfft(result, n=first_n, axis=first_ax)

    # ------------------------------------------------------------------
    # Automatic differentiation  —  GradientTape-based
    # ------------------------------------------------------------------

    def jvp(
        self,
        func: Callable[..., "tf.Tensor"],
        inputs: "tf.Tensor",
        tangents: "tf.Tensor",
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Native forward-mode AD via ``tf.autodiff.ForwardAccumulator``."""
        x = tf.convert_to_tensor(inputs, dtype=self._dtype)
        t = tf.convert_to_tensor(tangents, dtype=self._dtype)
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=t) as acc:
            value = func(x)
        tangent_out = acc.jvp(value)
        return value, tangent_out

    def _grad_forward(
        self,
        func: Callable[..., "tf.Tensor"],
        argnums_list: Sequence[int],
    ) -> Callable[..., "tf.Tensor"]:
        """Sweeps one-hot tangents via ForwardAccumulator — see torch/jax backends."""
        @functools.wraps(func)
        def grad_fn(*args: Any, **kwargs: Any) -> Any:
            modified = list(args)
            outs = []
            for idx in argnums_list:
                x = tf.convert_to_tensor(modified[idx], dtype=self._dtype)
                n_dims = x.shape[-1]

                def f_single(xi, _idx=idx):
                    call_args = list(modified)
                    call_args[_idx] = xi
                    return func(*call_args, **kwargs)

                cols = []
                for d in range(n_dims):
                    tangent = tf.one_hot(d, n_dims, dtype=self._dtype)
                    tangent = tf.broadcast_to(tangent, tf.shape(x))
                    _, jv = self.jvp(f_single, x, tangent)
                    cols.append(jv[..., None] if jv.shape.rank == x.shape.rank - 1 else jv)
                outs.append(tf.concat(cols, axis=-1) if cols[0].shape.rank else tf.stack(cols, axis=-1))
            return outs[0] if len(outs) == 1 else tuple(outs)

        return grad_fn

    def grad(
        self,
        func: Callable[..., "tf.Tensor"],
        argnums: Union[int, Sequence[int]] = 0,
        *,
        mode: str = "reverse",
    ) -> Callable[..., "tf.Tensor"]:
        """
        Returns a gradient function.

        ``mode="reverse"`` (default) uses nested persistent GradientTapes.
        The outer tape (``persistent=True``) records forward ops so that
        ``tape.gradient`` can be called multiple times for different outputs,
        and so that second-order calls (Hessians, Laplacians) work by nesting
        another call to ``grad`` on top.

        ``mode="forward"`` sweeps a ``ForwardAccumulator`` over each input
        dimension instead — cheaper when ``func`` has many output components
        relative to input width (coupled/mixed residual systems).

        TF has no first-class Taylor-mode/jet primitive, so ``mode="taylor"``
        composes two forward sweeps (forward-over-forward) via nested
        ``ForwardAccumulator``s rather than raising, at roughly 2x the cost
        of ``mode="forward"``.
        """
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        if mode == "forward":
            return self._grad_forward(func, argnums_list)
        if mode == "taylor":
            first = self._grad_forward(func, argnums_list)
            return self._grad_forward(first, argnums_list)
        if mode != "reverse":
            raise ValueError(f"Unknown differentiation mode: {mode!r}")

        @functools.wraps(func)
        def grad_fn(*args: Any, **kwargs: Any) -> Any:
            watched = [
                tf.Variable(tf.identity(args[i]), trainable=True, dtype=self._dtype)
                if not isinstance(args[i], tf.Variable)
                else args[i]
                for i in argnums_list
            ]
            modified = list(args)
            for idx, var in zip(argnums_list, watched):
                modified[idx] = var

            with tf.GradientTape(persistent=True) as tape:
                for var in watched:
                    tape.watch(var)
                output = func(*modified, **kwargs)
                scalar = (
                    output
                    if (output.shape.rank == 0 or output.shape.rank is None)
                    else tf.reduce_sum(output)
                )

            grads = [
                tape.gradient(scalar, var) for var in watched
            ]
            # Replace None with zeros
            grads = [
                g if g is not None else tf.zeros_like(w)
                for g, w in zip(grads, watched)
            ]
            del tape
            return grads[0] if isinstance(argnums, int) else tuple(grads)

        return grad_fn

    def value_and_grad(
        self,
        func: Callable[..., "tf.Tensor"],
        argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple["tf.Tensor", Any]]:
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        @functools.wraps(func)
        def vg_fn(*args: Any, **kwargs: Any) -> Tuple["tf.Tensor", Any]:
            watched = [
                tf.Variable(tf.identity(args[i]), trainable=True, dtype=self._dtype)
                if not isinstance(args[i], tf.Variable)
                else args[i]
                for i in argnums_list
            ]
            modified = list(args)
            for idx, var in zip(argnums_list, watched):
                modified[idx] = var

            with tf.GradientTape(persistent=True) as tape:
                for var in watched:
                    tape.watch(var)
                value  = func(*modified, **kwargs)
                scalar = (
                    value
                    if (value.shape.rank == 0 or value.shape.rank is None)
                    else tf.reduce_sum(value)
                )

            grads = [tape.gradient(scalar, var) for var in watched]
            grads = [
                g if g is not None else tf.zeros_like(w)
                for g, w in zip(grads, watched)
            ]
            del tape
            g_out = grads[0] if isinstance(argnums, int) else tuple(grads)
            return value, g_out

        return vg_fn

    def jacobian(
        self,
        func: Callable[..., "tf.Tensor"],
        inputs: "tf.Tensor",
    ) -> "tf.Tensor":
        # Use tape.watch() on the tensor directly rather than wrapping it
        # in a fresh tf.Variable — see the note in grad() above. Variable
        # assignment is non-differentiable, so wrapping an already-tracked
        # tensor (e.g. the output of an outer tape/grad call) here would
        # silently sever gradient flow back to the true input.
        x = tf.convert_to_tensor(inputs, dtype=self._dtype)
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = func(x)
        return tape.jacobian(output, x)

    def hessian(
        self,
        func: Callable[..., "tf.Tensor"],
        inputs: "tf.Tensor",
    ) -> "tf.Tensor":
        x = tf.convert_to_tensor(inputs, dtype=self._dtype)
        with tf.GradientTape() as outer:
            outer.watch(x)
            with tf.GradientTape() as inner:
                inner.watch(x)
                output = func(x)
                scalar = tf.reduce_sum(output)
            grad = inner.gradient(scalar, x)
        return outer.jacobian(grad, x)

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
    ) -> "tf.keras.Model":
        return _build_keras_mlp(
            layer_sizes  = list(layer_sizes),
            activation   = activation,
            use_residual = use_residual,
            dtype        = dtype or self._dtype,
        )

    def parameters(self, model: "tf.keras.Model") -> List["tf.Variable"]:
        """
        Return all trainable parameters as genuine ``tf.Variable`` objects.

        Under TF >= 2.16 (bundled Keras 3), ``model.trainable_variables``
        returns Keras's own backend-agnostic ``Variable`` wrapper rather
        than a raw ``tf.Variable`` — training still works transparently
        (``optimizer.apply_gradients`` understands both), but anything
        that expects a genuine ``tf.Variable`` (including this library's
        own tests) sees the wrong type. Unwrap to the underlying
        ``tf.Variable`` when running under Keras 3.
        """
        variables = model.trainable_variables
        unwrapped: List["tf.Variable"] = []
        for v in variables:
            if isinstance(v, tf.Variable):
                unwrapped.append(v)
                continue
            # Try the known internal attribute names Keras 3's TF-backend
            # Variable wrapper has used to hold the real tf.Variable.
            raw = None
            for attr in ("_value", "value"):
                candidate = getattr(v, attr, None)
                if isinstance(candidate, tf.Variable):
                    raw = candidate
                    break
            unwrapped.append(raw if raw is not None else v)
        return unwrapped

    def zero_grad(self, model: "tf.keras.Model") -> None:
        pass  # TF accumulates no persistent grads outside a tape

    # ------------------------------------------------------------------
    # Optimiser helpers
    # ------------------------------------------------------------------

    def build_optimizer(
        self,
        model: "tf.keras.Model",
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> "tf.keras.optimizers.Optimizer":
        if optimizer_name.lower() not in {"adamw", "adamax", "lamb"}:
            kwargs.pop("weight_decay", None) 
        return _build_tf_optimizer(optimizer_name, lr, **kwargs)

    def optimizer_step(
        self,
        optimizer: "tf.keras.optimizers.Optimizer",
        loss: "tf.Tensor",
        *,
        model: Optional["tf.keras.Model"] = None,
        tape: Optional["tf.GradientTape"] = None,
    ) -> None:
        """
        Applies one gradient update.

        When called with a pre-existing ``tape`` (recommended pattern for
        physics residual training loops), gradients are extracted from it.
        Otherwise, a new tape is created internally — useful for simple cases.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
        loss      : scalar loss tensor
        model     : Keras model; required to access trainable_variables
        tape      : an already-open GradientTape that recorded the loss
        """
        if model is None:
            raise ValueError(
                "TensorFlowBackend.optimizer_step requires 'model' to access "
                "trainable_variables."
            )

        if tape is None:
            raise ValueError(
                "TensorFlowBackend.optimizer_step requires an already-open "
                "'tape' that recorded the forward pass producing 'loss'. "
                "A tensor computed outside of any tape carries no gradient "
                "information, so there is no way to recompute gradients "
                "for it after the fact — re-run the forward pass inside a "
                "tf.GradientTape() block and pass that tape in explicitly."
            )

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # ------------------------------------------------------------------
    # JIT
    # ------------------------------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        """Wraps ``func`` in ``tf.function`` for graph-mode compilation."""
        return tf.function(func, **kwargs)

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, Sequence[Optional[int]]] = 0,
        out_axes: int = 0,
    ) -> Callable:
        """
        TensorFlow does not have a true ``vmap``.
        Uses ``tf.vectorized_map`` for simple element-wise cases.

        ``in_axes``/``out_axes`` are honored (not silently dropped): each
        input is moved so its mapped axis sits at 0 before mapping (which
        is what ``tf.vectorized_map`` requires), and the output is moved
        back to ``out_axes`` afterward. ``func`` is called with unpacked
        positional arguments, matching the calling convention of the
        JAX/PyTorch backends' ``vmap`` — not a single tuple argument,
        which is what plain ``tf.vectorized_map(func, args)`` would hand
        it and which breaks any ``func`` written against that convention.
        """
        @functools.wraps(func)
        def vmapped(*args: Any) -> "tf.Tensor":
            n = len(args)
            axes_list = (
                [in_axes] * n if isinstance(in_axes, int) else list(in_axes)
            )
            if len(axes_list) != n:
                raise ValueError(
                    f"in_axes has {len(axes_list)} entries but {n} "
                    f"positional arguments were passed."
                )

            moved = []
            for a, ax in zip(args, axes_list):
                if ax is None:
                    raise ValueError(
                        "TensorFlowBackend.vmap does not support "
                        "in_axes=None (broadcasting an unmapped argument); "
                        "every argument must have a mapped axis."
                    )
                moved.append(a if ax == 0 else tf.experimental.numpy.moveaxis(a, ax, 0))

            def _unpacked(elems: Tuple[Any, ...]) -> Any:
                return func(*elems)

            result = tf.vectorized_map(_unpacked, tuple(moved))

            if out_axes != 0:
                result = tf.experimental.numpy.moveaxis(result, 0, out_axes)
            return result

        return vmapped

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def default_device(self) -> str:
        return self._device

    def to_device(self, tensor: "tf.Tensor", device: str) -> "tf.Tensor":
        with tf.device(device):
            return tf.identity(tensor)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            self._device = kwargs["device"]
        if "mixed_precision" in kwargs and kwargs["mixed_precision"]:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

    def seed(self, value: int) -> None:
        tf.random.set_seed(value)