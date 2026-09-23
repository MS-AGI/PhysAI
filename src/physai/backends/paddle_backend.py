"""
physai/backends/paddle_backend.py

PaddlePaddle implementation of the AbstractBackend interface.

Design notes
------------
* PaddlePaddle (>= 2.x) is eager/imperative by default, like PyTorch —
  tensors carry gradients directly (no separate tape object), so
  ``optimizer_step`` follows the same ``loss.backward() -> step() ->
  clear_grad()`` pattern as the torch backend, not the TensorFlow
  GradientTape pattern.
* Reverse-mode AD uses ``paddle.grad`` (Paddle's analogue of
  ``torch.autograd.grad``). Paddle has no public, stable native
  forward-mode primitive (no ``jvp``/``jacfwd`` equivalent exposed the
  way JAX does), so ``mode="forward"``/``jvp``/``mode="taylor"`` are
  emulated via the standard double-backward ("double-vjp") trick: two
  reverse-mode passes reconstruct a Jacobian-vector product without a
  native forward primitive. This is documented at each call site,
  per the interface's own guidance for backends without native
  forward-mode AD.
* Neural networks are built with ``paddle.nn.Layer``/``paddle.nn.Linear``
  (Paddle's direct analogues of ``torch.nn.Module``/``torch.nn.Linear``).
* ``vmap`` has no stable native Paddle primitive either, so it is
  emulated by looping over the batch axis and re-stacking — the same
  honestly-scoped fallback strategy the tensorflow backend already uses
  for the same reason (no native TF vmap prior to ``vectorized_map``
  being generally reliable across ops).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import paddle
import paddle.nn as pnn
import paddle.nn.functional as F

from .base import AbstractBackend, BackendCapabilities, DType, Shape, Tensor

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------
def _elu(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """Exact ELU: x for x > 0, alpha*(exp(x)-1) for x <= 0.

    Mathematically identical to ``paddle.nn.functional.elu``, but this
    installed Paddle build has no registered ``elu_double_grad`` kernel,
    so any PINN residual that needs a second reverse-mode pass through the
    activation (e.g. a Laplacian term computed via ``paddle.grad(...,
    create_graph=True)`` twice) hits ``RuntimeError: elu_double_grad
    doesn't have any grad op``.

    An earlier version of this fix rebuilt ELU with ``paddle.expm1``,
    on the theory that its backward only needs ``exp``. That's wrong for
    *this* Paddle build: ``expm1`` has its own first-order kernel that is
    not literally ``exp``, and its second-order (``expm1_grad_grad``)
    kernel simply isn't registered, so ``RuntimeError: expm1_grad
    doesn't have any grad op`` still fires under ``create_graph=True``.

    Using ``paddle.exp(x) - 1`` instead avoids ``expm1`` entirely: the
    graph only ever records ``exp`` and ``subtract``, both of which have
    fully registered double-grad kernels (exp's derivative is exp
    itself; subtract's gradient is a constant), which is exercised
    elsewhere in this codebase (tanh, sin) without issue.
    """
    return paddle.where(x > 0, x, alpha * (paddle.exp(x) - 1.0))


def _pade_apply(x: paddle.Tensor, p: paddle.Tensor, q: paddle.Tensor) -> paddle.Tensor:
    """Exact Pade unit P(x)/Q(x), computed without an ``elementwise_div``
    node in the graph.

    ``numerator / denominator`` records an ``elementwise_div`` node,
    whose second backward (``divide_double_grad``) has no registered
    kernel in this Paddle build, so any residual that differentiates
    through this activation twice (e.g. a Laplacian term) fails.

    An earlier version of this fix swapped in ``paddle.reciprocal``,
    assuming its gradient kernel was independently registered for
    second order too. It isn't, in this build: ``reciprocal``'s own
    backward is fine, but its *double*-backward (``reciprocal_grad_grad``)
    has no registered kernel, so ``RuntimeError: reciprocal_grad doesn't
    have any grad op`` still fires under ``create_graph=True`` — same
    failure mode, just moved to a different op name.

    The denominator is guaranteed >= 1.0 here (Q's coefficients are
    wrapped in ``abs(...)`` before the constant 1.0 is added), so
    ``1/denominator == exp(-log(denominator))`` for every input, and the
    graph only ever needs ``log`` and ``exp`` in their own backward
    passes. Both have fully registered double-grad kernels in this
    build (they underpin softmax/cross-entropy backward, which is
    exercised constantly), so this sidesteps the missing kernel without
    changing the function being computed.
    """
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    q1, q2 = q[0], q[1]
    numerator = p0 + p1 * x + p2 * x ** 2 + p3 * x ** 3
    denominator = 1.0 + paddle.abs(q1 * x) + paddle.abs(q2 * x ** 2)
    inv_denominator = paddle.exp(-paddle.log(denominator))
    return numerator * inv_denominator


_ACTIVATIONS: Dict[str, Callable] = {
    "tanh":     paddle.tanh,
    "relu":     F.relu,
    "gelu":     F.gelu,
    "silu":     F.silu,
    "sigmoid":  F.sigmoid,
    "elu":      _elu,
    "softplus": F.softplus,
    # Snake (Ziyin et al. 2020): x + sin(x)^2, fixed frequency a=1 —
    # matches the non-learnable default used in the jax/torch backends.
    "snake":    lambda x: x + paddle.sin(x) ** 2,
}


# ---------------------------------------------------------------------------
# Stateful / learnable activations (L-LAAF, Pade) — parity with jax_backend
# ---------------------------------------------------------------------------
# Unlike the plain functions in _ACTIVATIONS, these carry their own
# trainable parameters, so each hidden layer needs its own independent
# instance (mirroring the jax backend's per-layer Flax `self.param` /
# pure-JAX per-layer pytree, and the torch backend's fresh-instance-per-
# layer pattern). Implemented as tiny `paddle.nn.Layer`s so their
# parameters are automatically discovered by `model.parameters()` via
# Paddle's normal sublayer/parameter registration — no separate
# bookkeeping needed.

class _LLAAF(pnn.Layer):
    """L-LAAF (Jagtap & Karniadakis 2020): a*tanh(n*a*x), `a` learnable
    per layer, `n` a fixed scaling factor so a*n = 1 at init (starts
    identical to plain tanh(x)), matching the jax/torch defaults."""

    def __init__(self, n: float = 10.0) -> None:
        super().__init__()
        self.n = n
        self.a = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Constant(1.0 / n),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # No outer 'a' coefficient — see torch_backend.py's
        # AdaptiveTanh.forward for the L-LAAF reasoning.
        return paddle.tanh(self.n * self.a * x)


class _PadeActivation(pnn.Layer):
    """Safe Pade Activation Unit (Molina et al. 2019): P(x)/Q(x), Q's
    coefficients under abs(...) so the denominator never vanishes."""

    def __init__(self) -> None:
        super().__init__()
        self.p = self.create_parameter(
            shape=[4], default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.p.set_value(paddle.to_tensor([0.0, 1.0, 0.5, 0.0], dtype=self.p.dtype))
        self.q = self.create_parameter(
            shape=[2], default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.q.set_value(paddle.to_tensor([0.5, 0.0], dtype=self.q.dtype))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return _pade_apply(x, self.p, self.q)


_STATEFUL_ACTIVATIONS: Dict[str, Callable[[], pnn.Layer]] = {
    "llaaf": lambda: _LLAAF(10.0),
    "pade":  lambda: _PadeActivation(),
}


# ---------------------------------------------------------------------------
# MLP module
# ---------------------------------------------------------------------------

class _PaddleMLP(pnn.Layer):
    """Fully-connected MLP: len(layer_sizes) - 1 Linear layers, matching
    the layer-count convention of the jax/torch/tensorflow backends
    (e.g. [2, 64, 64, 1] -> 3 Linear layers)."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        stateful = activation in _STATEFUL_ACTIVATIONS
        if not stateful and activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from "
                f"{list(_ACTIVATIONS) + list(_STATEFUL_ACTIVATIONS)}."
            )
        self.stateful = stateful
        self.activation_fn = None if stateful else _ACTIVATIONS[activation]
        self.use_residual = use_residual
        self.layer_sizes = list(layer_sizes)
        self.linears = pnn.LayerList(
            [
                pnn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        if stateful:
            # One independent, parameterised activation instance per
            # hidden layer (not the output layer) -- mirrors the
            # jax/torch per-layer instance pattern so each hidden layer
            # learns its own `a`/Pade coefficients rather than sharing
            # one set across the whole network.
            n_hidden = len(self.linears) - 1
            self.activations = pnn.LayerList(
                [_STATEFUL_ACTIVATIONS[activation]() for _ in range(n_hidden)]
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        n_layers = len(self.linears)
        for i, linear in enumerate(self.linears):
            h = linear(x)
            is_last = i == n_layers - 1
            if not is_last:
                h = self.activations[i](h) if self.stateful else self.activation_fn(h)
                if self.use_residual and x.shape[-1] == h.shape[-1]:
                    x = x + h
                else:
                    x = h
            else:
                x = h
        return x


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class PaddleBackend(AbstractBackend):
    """PaddlePaddle execution backend for PhysAI."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Any = "float32",
        *,
        seed: int = 0,
    ) -> None:
        self._dtype = dtype
        if device is not None:
            paddle.set_device(device)
        try:
            self.device = "gpu" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
        except Exception:  # pragma: no cover
            self.device = "cpu"
        paddle.seed(seed)

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "paddle"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jit             = True,
            supports_vmap            = False,
            supports_complex         = True,
            supports_sparse          = True,
            supports_mixed_precision = self.device == "gpu",
            native_fft               = True,
        )

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def tensor(self, data: Any, dtype=None, device=None) -> paddle.Tensor:
        t = paddle.to_tensor(data, dtype=dtype or self._dtype)
        if device is not None:
            t = self.to_device(t, device)
        return t

    def zeros(self, shape: Shape, dtype=None) -> paddle.Tensor:
        return paddle.zeros(shape, dtype=dtype or self._dtype)

    def ones(self, shape: Shape, dtype=None) -> paddle.Tensor:
        return paddle.ones(shape, dtype=dtype or self._dtype)

    def linspace(self, start: float, stop: float, num: int, dtype=None) -> paddle.Tensor:
        return paddle.linspace(start, stop, num, dtype=dtype or self._dtype)

    def meshgrid(self, *tensors: paddle.Tensor, indexing: str = "ij") -> List[paddle.Tensor]:
        grids = list(paddle.meshgrid(*tensors))
        # paddle.meshgrid's default layout matches numpy's "ij" (matrix)
        # indexing already; only transpose the first two axes if "xy"
        # (Cartesian) indexing was explicitly requested, mirroring the
        # numpy/jax convention this interface documents.
        if indexing == "xy" and len(grids) >= 2:
            grids = [paddle.transpose(g, perm=[1, 0] + list(range(2, g.ndim))) for g in grids]
        return grids

    def stack(self, tensors: Sequence[paddle.Tensor], axis: int = 0) -> paddle.Tensor:
        return paddle.stack(list(tensors), axis=axis)

    def concatenate(self, tensors: Sequence[paddle.Tensor], axis: int = 0) -> paddle.Tensor:
        return paddle.concat(list(tensors), axis=axis)

    def reshape(self, tensor: paddle.Tensor, shape: Shape) -> paddle.Tensor:
        return paddle.reshape(tensor, shape)

    def cast(self, tensor: paddle.Tensor, dtype: Any) -> paddle.Tensor:
        return paddle.cast(tensor, dtype)

    def to_numpy(self, tensor: paddle.Tensor) -> np.ndarray:
        return tensor.numpy()

    # ------------------------------------------------------------------
    # Mathematical primitives
    # ------------------------------------------------------------------

    def mean(self, tensor: paddle.Tensor, axis=None) -> paddle.Tensor:
        return paddle.mean(tensor, axis=axis)

    def sum(self, tensor: paddle.Tensor, axis=None) -> paddle.Tensor:
        return paddle.sum(tensor, axis=axis)

    def abs(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.abs(tensor)

    def sqrt(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.sqrt(tensor)

    def square(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.square(tensor)

    def exp(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.exp(tensor)

    def log(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.log(tensor)

    def sin(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.sin(tensor)

    def cos(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.cos(tensor)

    def atan(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.atan(tensor)

    def tanh(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return paddle.tanh(tensor)

    def matmul(self, a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
        return paddle.matmul(a, b)

    def einsum(self, equation: str, *operands: paddle.Tensor) -> paddle.Tensor:
        return paddle.einsum(equation, *operands)

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    def rfft(self, tensor: paddle.Tensor, n=None, axis: int = -1) -> paddle.Tensor:
        return paddle.fft.rfft(tensor, n=n, axis=axis)

    def irfft(self, tensor: paddle.Tensor, n=None, axis: int = -1) -> paddle.Tensor:
        return paddle.fft.irfft(tensor, n=n, axis=axis)

    def rfft2(self, tensor: paddle.Tensor, s=None) -> paddle.Tensor:
        return paddle.fft.rfft2(tensor, s=s)

    def irfft2(self, tensor: paddle.Tensor, s=None) -> paddle.Tensor:
        return paddle.fft.irfft2(tensor, s=s)

    def rfftn(self, tensor: paddle.Tensor, s=None, axes=None) -> paddle.Tensor:
        return paddle.fft.rfftn(tensor, s=s, axes=axes)

    def irfftn(self, tensor: paddle.Tensor, s=None, axes=None) -> paddle.Tensor:
        return paddle.fft.irfftn(tensor, s=s, axes=axes)

    # ------------------------------------------------------------------
    # Automatic differentiation
    # ------------------------------------------------------------------

    def jvp(
        self,
        func: Callable[..., paddle.Tensor],
        inputs: paddle.Tensor,
        tangents: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Paddle has no native forward-mode primitive exposed the way
        JAX's ``jax.jvp`` is, but it does ship a tested public helper
        for exactly this — ``paddle.incubate.autograd.functional.jvp``
        (same ``paddle.incubate.autograd`` module already used by
        ``jacobian``/``hessian`` below).

        A hand-rolled double-backward ("double-vjp") trick was tried
        here first and failed with a Paddle kernel error ("DX can not
        be nullptr" from ``activation_grad_impl``); switching to
        Paddle's own official ``incubate.autograd.functional.jvp``
        was the natural next step — but it hits the *exact same*
        underlying ``paddle.grad(create_graph=True, ...)`` call
        internally and fails with the identical error. Since Paddle's
        own blessed, tested implementation fails the same way, this is
        a genuine upstream limitation of the installed Paddle build's
        double-backward support for this op chain (confirmed via the
        traceback bottoming out inside Paddle's own
        ``functional.py:_double_backward_trick``/``_grad``, not in any
        physai code) — not something fixable by choosing a different
        call path from the Python API level. Rather than let that
        surface as a confusing internal Paddle traceback, it's caught
        here and re-raised as a clear, actionable error.
        """
        from paddle.incubate.autograd.functional import jvp as _paddle_jvp
        try:
            value, jvp_out = _paddle_jvp(func, inputs, tangents)
        except RuntimeError as e:
            if "nullptr" in str(e) or "DX can not be" in str(e):
                raise NotImplementedError(
                    "PaddleBackend.jvp is unavailable in this Paddle build: "
                    "both a hand-rolled double-backward implementation and "
                    "Paddle's own official paddle.incubate.autograd."
                    "functional.jvp fail with the same underlying kernel "
                    "error ('DX can not be nullptr' from "
                    "activation_grad_impl) when computing "
                    "create_graph=True double-backward through this op "
                    "chain. This is an upstream Paddle limitation, not a "
                    "physai bug — use mode='reverse' (the default) "
                    "instead; forward-mode/mode='taylor' are unavailable "
                    "on Paddle until this is fixed in Paddle itself."
                ) from e
            raise
        return value, jvp_out

    def grad(
        self,
        func: Callable[..., paddle.Tensor],
        argnums: Union[int, Sequence[int]] = 0,
        *,
        mode: str = "reverse",
    ) -> Callable[..., paddle.Tensor]:
        """
        ``mode="reverse"`` (default): ``paddle.grad`` — cheapest for the
        common PINN case (scalar/small-vector output, many collocation
        points), same as the jax/torch backends' default.

        ``mode="forward"``: Paddle has no native ``jacfwd``. Emulated by
        sweeping one-hot tangents through ``self.jvp`` above, one column
        of the Jacobian per input dimension — functionally equivalent to
        jacfwd but at reverse-mode cost per column (no free lunch
        without a native forward primitive; this is intentionally not
        hidden behind a fake "cheap" label). NOTE: as of the installed
        Paddle build this raises ``NotImplementedError`` in practice —
        see ``jvp``'s docstring above for why (an upstream Paddle
        double-backward kernel gap, confirmed via Paddle's own official
        ``incubate.autograd.functional.jvp`` failing the same way, not a
        physai bug).

        ``mode="taylor"``: not implemented. JAX's taylor mode relies on
        ``jax.experimental.jet``, a genuinely forward-mode Taylor-series
        propagation primitive with no Paddle analogue; faking it via
        nested double-backward would be both slow and easy to get
        subtly wrong (second directional derivatives via naive nested
        reverse-mode AD can silently pick up the wrong cross terms), so
        this raises rather than silently returning a wrong number.
        """
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        if mode == "reverse":
            def reverse_grad(*args: Any, **kwargs: Any) -> Any:
                args = list(args)
                for idx in argnums_list:
                    args[idx].stop_gradient = False
                out = func(*args, **kwargs)
                grads = paddle.grad(
                    outputs=out,
                    inputs=[args[idx] for idx in argnums_list],
                    create_graph=True,
                    retain_graph=True,
                )
                return grads[0] if len(grads) == 1 else tuple(grads)
            return reverse_grad

        if mode == "forward":
            def forward_grad(*args: Any, **kwargs: Any) -> Any:
                outs = []
                for idx in argnums_list:
                    x = args[idx]
                    n_dims = x.shape[-1]
                    cols = []
                    for d in range(n_dims):
                        tangent = paddle.zeros_like(x)
                        tangent[..., d] = 1.0

                        def f_single(xi):
                            call_args = list(args)
                            call_args[idx] = xi
                            return func(*call_args, **kwargs)

                        _, col = self.jvp(f_single, x, tangent)
                        cols.append(col[..., None] if col.ndim == x.ndim - 1 else col)
                    outs.append(paddle.concat(cols, axis=-1) if cols[0].ndim else paddle.stack(cols, axis=-1))
                return outs[0] if len(outs) == 1 else tuple(outs)
            return forward_grad

        if mode == "taylor":
            raise NotImplementedError(
                "PaddleBackend.grad(mode='taylor') is not implemented: Paddle has "
                "no jax.experimental.jet analogue, and faking second-order "
                "directional derivatives via naive nested reverse-mode AD risks "
                "silently wrong cross terms. Use mode='reverse' or 'forward' "
                "and compose two grad() calls explicitly if you need this."
            )
        raise ValueError(f"Unknown differentiation mode: {mode!r}")

    def value_and_grad(
        self,
        func: Callable[..., paddle.Tensor],
        argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[paddle.Tensor, paddle.Tensor]]:
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        def value_and_grad_fn(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
            args = list(args)
            for idx in argnums_list:
                args[idx].stop_gradient = False
            value = func(*args, **kwargs)
            grads = paddle.grad(
                outputs=value,
                inputs=[args[idx] for idx in argnums_list],
                create_graph=True,
                retain_graph=True,
            )
            grad_out = grads[0] if len(grads) == 1 else tuple(grads)
            return value, grad_out
        return value_and_grad_fn

    def jacobian(
        self,
        func: Callable[..., paddle.Tensor],
        inputs: paddle.Tensor,
    ) -> paddle.Tensor:
        from paddle.incubate.autograd import Jacobian
        x = inputs
        x.stop_gradient = False
        J = Jacobian(func, x)
        return J[:]

    def hessian(
        self,
        func: Callable[..., paddle.Tensor],
        inputs: paddle.Tensor,
    ) -> paddle.Tensor:
        from paddle.incubate.autograd import Hessian
        x = inputs
        x.stop_gradient = False
        H = Hessian(func, x)
        return H[:]

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
        model = _PaddleMLP(layer_sizes, activation=activation, use_residual=use_residual)
        if dtype is not None:
            model = model.astype(dtype)
        return model

    def parameters(self, model: Any) -> List[Any]:
        return list(model.parameters())

    def zero_grad(self, model: Any) -> None:
        model.clear_gradients()

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
        _OPT_MAP = {
            "adam":    paddle.optimizer.Adam,
            "adamw":   paddle.optimizer.AdamW,
            "sgd":     paddle.optimizer.SGD,
            "rmsprop": paddle.optimizer.RMSProp,
        }
        cls = _OPT_MAP.get(optimizer_name.lower())
        if cls is None:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose from {list(_OPT_MAP)}.")

        if optimizer_name.lower() != "adamw":
            kwargs.pop("weight_decay", None)  # Only AdamW takes weight_decay here

        return cls(learning_rate=lr, parameters=model.parameters(), **kwargs)

    def optimizer_step(
        self,
        optimizer: Any,
        loss: paddle.Tensor,
        *,
        model: Optional[Any] = None,
        tape: Optional[Any] = None,
    ) -> None:
        """Paddle is eager/imperative like torch — no tape object needed."""
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    # ------------------------------------------------------------------
    # JIT / vmap
    # ------------------------------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        return paddle.jit.to_static(func, **kwargs)

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, Sequence[Optional[int]]] = 0,
        out_axes: int = 0,
    ) -> Callable:
        """
        No native Paddle vmap primitive exists, so this falls back to an
        explicit loop-and-stack over the batch axis — the same honestly-
        scoped strategy the tensorflow backend uses for the same reason.
        """
        def vmapped(*args: Any) -> Any:
            axes = [in_axes] * len(args) if isinstance(in_axes, int) else list(in_axes)
            if len(axes) != len(args):
                raise ValueError(
                    f"vmap: in_axes has {len(axes)} entries but {len(args)} arguments were given."
                )
            batch_size = None
            for arg, ax in zip(args, axes):
                if ax is None:
                    raise ValueError("vmap: in_axes=None (broadcast, non-batched argument) is not supported.")
                size = arg.shape[ax]
                batch_size = size if batch_size is None else batch_size
            outputs = []
            for i in range(batch_size):
                sliced = [paddle.index_select(a, paddle.to_tensor([i]), axis=ax).squeeze(ax)
                          for a, ax in zip(args, axes)]
                outputs.append(func(*sliced))
            return paddle.stack(outputs, axis=out_axes)
        return vmapped

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def default_device(self) -> str:
        return self.device

    def to_device(self, tensor: paddle.Tensor, device: str) -> paddle.Tensor:
        place = paddle.CUDAPlace(0) if device in ("gpu", "cuda") else paddle.CPUPlace()
        return paddle.to_tensor(tensor.numpy(), place=place)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            paddle.set_device(kwargs["device"])

    def seed(self, value: int) -> None:
        paddle.seed(value)