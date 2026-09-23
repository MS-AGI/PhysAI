"""
physai/backends/torch_backend.py

PyTorch implementation of the AbstractBackend interface.

Design notes
------------
* Gradients are computed via ``torch.autograd.grad`` — we never call
  ``.detach()`` during a physics residual pass so that the computation
  graph stays intact for higher-order derivatives.
* ``create_graph=True`` is the default so that second-order derivatives
  (Hessian, Laplacian) work correctly out of the box.
* JIT is exposed via ``torch.compile`` (PyTorch ≥ 2.0) with a graceful
  fallback to ``torch.jit.script``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.fft as torch_fft

from .base import AbstractBackend, BackendCapabilities, DType, Shape, Tensor

import numpy as np  # type: ignore
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class Snake(nn.Module):
    """
    Snake activation (Ziyin et al. 2020): x + (1/a)*sin(a*x)^2.

    A periodic-plus-linear activation for convection-dominated / shock
    -forming residuals (Burgers, Navier-Stokes, Euler, ...) that captures
    oscillatory structure without SIREN's brittle initialization
    requirements -- unlike raw sin(x), it degrades gracefully toward the
    identity as `a` -> 0 rather than needing a specific weight-init scheme
    to train at all.
    """
    def __init__(self, a: float = 1.0) -> None:
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2


class AdaptiveTanh(nn.Module):
    """
    Layer-wise Locally Adaptive Activation Function, L-LAAF
    (Jagtap & Karniadakis 2020): a * tanh(n * a * x), with `a` a
    trainable scalar (one per activation instance -- i.e. one per layer,
    since build_mlp below now instantiates a fresh activation per layer)
    and `n` a fixed scaling factor. Lets the network locally sharpen or
    flatten its own nonlinearity where a residual's gradient is steep,
    instead of using one fixed slope everywhere -- the PINN-specific fix
    this library's high-gradient/stiff activation buckets actually want,
    with documented convergence-speed gains over static activations on
    PINN benchmarks.
    """
    def __init__(self, n: float = 10.0) -> None:
        super().__init__()
        self.n = n
        # a*n = 1 at init, so this starts out identical to plain tanh(x)
        # and only diverges from it as training adapts `a`.
        self.a = nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Jagtap & Karniadakis (2020)'s L-LAAF has no outer coefficient —
        # it's tanh(n*a*x), scaling only the pre-activation argument.
        return torch.tanh(self.n * self.a * x)


class PadeActivation(nn.Module):
    """
    Safe Padé Activation Unit, PAU (Molina et al. 2019): a learnable
    rational function P(x)/Q(x), P degree 3 / Q degree 2, with Q's
    coefficients used inside abs(...) so the denominator is bounded away
    from zero everywhere -- the "safe" variant from the paper,
    guaranteeing no poles ever appear during training. Smooth,
    spectral-bias-resistant, and strictly more expressive than a fixed
    activation shape, at the cost of 6 extra learnable scalars per layer.
    Initialised close to a GELU-like curve.
    """
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([0.0, 1.0, 0.5, 0.0]))  # p0..p3
        self.q = nn.Parameter(torch.tensor([0.5, 0.0]))            # q1, q2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, p1, p2, p3 = self.p
        q1, q2 = self.q
        numerator = p0 + p1 * x + p2 * x ** 2 + p3 * x ** 3
        denominator = 1.0 + torch.abs(q1 * x) + torch.abs(q2 * x ** 2)
        return numerator / denominator


_ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "tanh":    nn.Tanh,
    "relu":    nn.ReLU,
    "gelu":    nn.GELU,
    "silu":    nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "elu":     nn.ELU,
    "mish":    nn.Mish,
    "softplus": nn.Softplus,
    "snake":   Snake,
    "llaaf":   AdaptiveTanh,
    "pade":    PadeActivation,
}

_OPTIMIZERS: Dict[str, type] = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "sgd":     torch.optim.SGD,
    "lbfgs":   torch.optim.LBFGS,
    "rmsprop": torch.optim.RMSprop,
}


class _ResidualLinear(nn.Module):
    """
    Single Linear + activation, with a residual add when shapes match:
    ``x = x + act(Linear(x))``.

    Matches the JAX (_FlaxMLP) and TF (_build_keras_mlp) residual layer
    exactly — one Dense/Linear per requested layer_sizes entry. The
    previous implementation here (_ResidualBlock) substituted a *two*-Linear
    block in place of a single layer whenever a residual connection
    applied, silently giving the torch backend a deeper network with
    roughly double the parameters of the JAX/TF models for the same
    layer_sizes + use_residual=True config — breaking the backend-agnostic
    architecture parity the library is built around.
    """

    def __init__(self, in_f: int, out_f: int, activation_cls: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        # Fresh instance per layer: matters for stateful/learnable
        # activations (Snake/L-LAAF/Pade), which must not share
        # parameters across layers the way a stateless nn.Tanh() safely
        # could.
        self.act = activation_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc(x))
        return x + h


class _MLP(nn.Module):
    """Fully-connected MLP, optionally with residual connections."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_cls: Callable[[], nn.Module],
        use_residual: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            is_last = i == len(layer_sizes) - 2
            if use_residual and not is_last and in_f == out_f:
                layers.append(_ResidualLinear(in_f, out_f, activation_cls))
            else:
                layers.append(nn.Linear(in_f, out_f))
                if not is_last:
                    # Fresh instance per layer -- see _ResidualLinear's
                    # comment; the previous version passed one
                    # pre-constructed activation instance and reused it
                    # across every layer, which is only safe for
                    # stateless activations.
                    layers.append(activation_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class TorchBackend(AbstractBackend):
    """PyTorch execution backend for PhysAI."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        *,
        enable_tf32: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._dtype  = dtype
        if enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "torch"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jit             = True,
            supports_vmap            = True,
            supports_complex         = True,
            supports_sparse          = True,
            supports_mixed_precision = torch.cuda.is_available(),
            native_fft               = True,
        )

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        dt  = dtype  or self._dtype
        dev = device or self._device
        return torch.as_tensor(data, dtype=dt, device=dev)

    def zeros(self, shape: Shape, dtype: Optional[DType] = None) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype or self._dtype, device=self._device)

    def ones(self, shape: Shape, dtype: Optional[DType] = None) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype or self._dtype, device=self._device)

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: Optional[DType] = None,
    ) -> torch.Tensor:
        return torch.linspace(start, stop, num, dtype=dtype or self._dtype, device=self._device)

    def meshgrid(self, *tensors: torch.Tensor, indexing: str = "ij") -> List[torch.Tensor]:
        return list(torch.meshgrid(*tensors, indexing=indexing))

    def stack(self, tensors: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.stack(list(tensors), dim=axis)

    def concatenate(self, tensors: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.cat(list(tensors), dim=axis)

    def reshape(self, tensor: torch.Tensor, shape: Shape) -> torch.Tensor:
        return tensor.reshape(shape)

    def cast(self, tensor: torch.Tensor, dtype: DType) -> torch.Tensor:
        return tensor.to(dtype=dtype)

    def to_numpy(self, tensor: torch.Tensor) -> "np.ndarray":  # noqa: F821
        import numpy as np
        return tensor.cpu().detach().numpy()

    # ------------------------------------------------------------------
    # Mathematical primitives
    # ------------------------------------------------------------------

    def mean(self, tensor: torch.Tensor, axis=None) -> torch.Tensor:
        return tensor.mean() if axis is None else tensor.mean(dim=axis)

    def sum(self, tensor: torch.Tensor, axis=None) -> torch.Tensor:
        return tensor.sum() if axis is None else tensor.sum(dim=axis)

    def abs(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.abs(tensor)

    def sqrt(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(tensor)

    def square(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** 2

    def exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(tensor)

    def log(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor)

    def sin(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(tensor)

    def cos(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cos(tensor)

    def atan(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.atan(tensor)

    def tanh(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(tensor)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    def einsum(self, equation: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(equation, *operands)

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    def rfft(self, tensor: torch.Tensor, n=None, axis: int = -1) -> torch.Tensor:
        return torch_fft.rfft(tensor, n=n, dim=axis)

    def irfft(self, tensor: torch.Tensor, n=None, axis: int = -1) -> torch.Tensor:
        return torch_fft.irfft(tensor, n=n, dim=axis)

    def rfft2(self, tensor: torch.Tensor, s=None) -> torch.Tensor:
        return torch_fft.rfft2(tensor, s=s)

    def irfft2(self, tensor: torch.Tensor, s=None) -> torch.Tensor:
        return torch_fft.irfft2(tensor, s=s)

    def rfftn(self, tensor: torch.Tensor, s=None, axes=None) -> torch.Tensor:
        return torch_fft.rfftn(tensor, s=s, dim=axes)

    def irfftn(self, tensor: torch.Tensor, s=None, axes=None) -> torch.Tensor:
        return torch_fft.irfftn(tensor, s=s, dim=axes)

    # ------------------------------------------------------------------
    # Automatic differentiation
    # ------------------------------------------------------------------

    def jvp(
        self,
        func: Callable[..., torch.Tensor],
        inputs: torch.Tensor,
        tangents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Native forward-mode AD via ``torch.func.jvp`` (dual numbers)."""
        from torch.func import jvp as _torch_jvp
        value, tangent_out = _torch_jvp(func, (inputs,), (tangents,))
        return value, tangent_out

    def _grad_forward(
        self,
        func: Callable[..., torch.Tensor],
        argnums_list: List[int],
    ) -> Callable[..., torch.Tensor]:
        """
        Forward-mode gradient: sweeps one-hot tangents over the last axis
        of each selected input and stacks the resulting JVPs. Cheapest when
        ``func`` has many output components relative to input width (e.g.
        coupled/mixed residual systems), since it needs one forward pass
        per input dimension rather than one reverse pass per output.
        """
        @functools.wraps(func)
        def grad_fn(*args: Any, **kwargs: Any) -> Any:
            modified = list(args)
            outs = []
            for idx in argnums_list:
                x = modified[idx]
                n_dims = x.shape[-1]

                def f_single(xi: torch.Tensor, _idx: int = idx) -> torch.Tensor:
                    call_args = list(modified)
                    call_args[_idx] = xi
                    return func(*call_args, **kwargs)

                cols = []
                for d in range(n_dims):
                    tangent = torch.zeros_like(x)
                    tangent[..., d] = 1.0
                    _, jv = self.jvp(f_single, x, tangent)
                    cols.append(jv[..., None] if jv.dim() == x.dim() - 1 else jv)
                outs.append(torch.cat(cols, dim=-1) if cols[0].dim() else torch.stack(cols, dim=-1))
            return outs[0] if len(outs) == 1 else tuple(outs)

        return grad_fn

    def _grad_taylor(
        self,
        func: Callable[..., torch.Tensor],
        argnums_list: List[int],
    ) -> Callable[..., torch.Tensor]:
        """
        Forward-over-forward ("Taylor-mode") directional second derivative:
        nests two JVPs so ∂²func/∂x_d² along each basis direction is
        obtained without a reverse-mode pass. Returns the same shape as
        ``_grad_forward`` but containing second derivatives — used by
        ``core.pde_residual`` for cheap Laplacian diagonals in low input
        dimension.
        """
        @functools.wraps(func)
        def grad2_fn(*args: Any, **kwargs: Any) -> Any:
            modified = list(args)
            outs = []
            for idx in argnums_list:
                x = modified[idx]
                n_dims = x.shape[-1]

                def f_single(xi: torch.Tensor, _idx: int = idx) -> torch.Tensor:
                    call_args = list(modified)
                    call_args[_idx] = xi
                    return func(*call_args, **kwargs)

                cols = []
                for d in range(n_dims):
                    tangent = torch.zeros_like(x)
                    tangent[..., d] = 1.0

                    def first_dir(xi: torch.Tensor, _t=tangent, _f=f_single) -> torch.Tensor:
                        _, jv = self.jvp(_f, xi, _t)
                        return jv

                    _, jv2 = self.jvp(first_dir, x, tangent)
                    cols.append(jv2[..., None] if jv2.dim() == x.dim() - 1 else jv2)
                outs.append(torch.cat(cols, dim=-1) if cols[0].dim() else torch.stack(cols, dim=-1))
            return outs[0] if len(outs) == 1 else tuple(outs)

        return grad2_fn

    def grad(
        self,
        func: Callable[..., torch.Tensor],
        argnums: Union[int, Sequence[int]] = 0,
        *,
        mode: str = "reverse",
    ) -> Callable[..., torch.Tensor]:
        """
        Returns a function computing ∂func/∂args[argnums].

        ``mode="reverse"`` (default) uses torch.autograd.grad with
        create_graph=True so that the gradient itself is differentiable
        (needed for Hessians, Laplacians, and physics residuals involving
        second-order PDE terms). ``mode="forward"``/``"taylor"`` dispatch to
        JVP-based sweeps — see ``_grad_forward``/``_grad_taylor``.
        """
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        if mode == "forward":
            return self._grad_forward(func, argnums_list)
        if mode == "taylor":
            return self._grad_taylor(func, argnums_list)
        if mode != "reverse":
            raise ValueError(f"Unknown differentiation mode: {mode!r}")

        @functools.wraps(func)
        def grad_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
            # Ensure selected inputs require gradients
            modified = list(args)
            for idx in argnums_list:
                t = modified[idx]
                if not t.requires_grad:
                    # FIX 1: Safely clone and detach tensors to create a true tracking leaf variable.
                    # This prevents crashes when "t" is an inline slice/view (e.g., points[:, 0:1]).
                    t = t.detach().clone().requires_grad_(True)
                    modified[idx] = t

            # FIX 2: Enforce gradient computation globally.
            # This explicitly overrides any outer 'with torch.no_grad():' blocks active during RAR generation.
            with torch.enable_grad():
                output = func(*modified, **kwargs)

                # FIX 4: If `output` itself doesn't require grad -- e.g. a
                # model that returns a tensor built entirely from constants
                # (backend.zeros(...)) with no computational link back to
                # `modified` at all -- torch.autograd.grad raises "element 0
                # of tensors does not require grad and does not have a
                # grad_fn" immediately, regardless of allow_unused=True.
                # allow_unused only tolerates *inputs* that a connected
                # output doesn't depend on; it does not tolerate an output
                # with no grad_fn whatsoever. Mathematically the correct
                # gradient of a genuine constant is exactly zero, so we
                # short-circuit to zero-filled gradients here instead of
                # crashing.
                if not output.requires_grad:
                    grads = tuple(
                        torch.zeros_like(modified[i]) for i in argnums_list
                    )
                    return grads[0] if isinstance(argnums, int) else grads

                # FIX 3: Using grad_outputs=torch.ones_like completely clean-reduces the vector
                # calculation back to its matrix size without relying on structural .sum() operations, 
                # resolving the "implicitly created only for scalar outputs" RuntimeError.
                grads = torch.autograd.grad(
                    outputs     = output,
                    inputs      = [modified[i] for i in argnums_list],
                    grad_outputs= torch.ones_like(output),
                    create_graph= True,
                    allow_unused= True,
                )

            # Replace None gradients with zeros
            grads = tuple(
                g if g is not None else torch.zeros_like(modified[i])
                for g, i in zip(grads, argnums_list)
            )
            return grads[0] if isinstance(argnums, int) else grads

        return grad_fn

    def value_and_grad(
        self,
        func: Callable[..., torch.Tensor],
        argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
        argnums_list = [argnums] if isinstance(argnums, int) else list(argnums)

        @functools.wraps(func)
        def vg_fn(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
            modified = list(args)
            for idx in argnums_list:
                t = modified[idx]
                if not t.requires_grad:
                    # Same fix as grad()'s FIX 1: clone+detach into a true
                    # leaf tensor before calling requires_grad_(True).
                    # Calling requires_grad_(True) directly on the
                    # original tensor (the old behavior here) crashes with
                    # "only leaf Tensors can have requires_grad_()" when t
                    # is a non-leaf slice/view (e.g. points[:, 0:1]), and
                    # even when it doesn't crash it mutates the caller's
                    # original tensor in place — leaving requires_grad=True
                    # on a tensor the caller may reuse elsewhere.
                    t = t.detach().clone().requires_grad_(True)
                    modified[idx] = t

            value = func(*modified, **kwargs)
            scalar = value if value.numel() == 1 else value.sum()
            grads  = torch.autograd.grad(
                outputs      = scalar,
                inputs       = [modified[i] for i in argnums_list],
                create_graph = True,
                allow_unused = True,
            )
            grads = tuple(
                g if g is not None else torch.zeros_like(modified[i])
                for g, i in zip(grads, argnums_list)
            )
            g_out = grads[0] if isinstance(argnums, int) else grads
            return value, g_out

        return vg_fn

    def jacobian(
        self,
        func: Callable[..., torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Full Jacobian via ``torch.autograd.functional.jacobian``."""
        return torch.autograd.functional.jacobian(func, inputs, create_graph=True)

    def hessian(
        self,
        func: Callable[..., torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Full Hessian via ``torch.autograd.functional.hessian``."""
        return torch.autograd.functional.hessian(func, inputs, create_graph=True)

    # ------------------------------------------------------------------
    # Neural-network model interface
    # ------------------------------------------------------------------

    def build_mlp(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        *,
        use_residual: bool = False,
        dtype: Optional[DType] = None,
    ) -> _MLP:
        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(_ACTIVATIONS)}."
            )
        model = _MLP(
            layer_sizes    = list(layer_sizes),
            activation_cls = act_cls,
            use_residual   = use_residual,
        )
        dt = dtype or self._dtype
        model = model.to(dtype=dt, device=self._device)
        return model

    def parameters(self, model: nn.Module) -> List[torch.Tensor]:
        return list(model.parameters())

    def zero_grad(self, model: nn.Module) -> None:
        model.zero_grad()

    # ------------------------------------------------------------------
    # Optimiser helpers
    # ------------------------------------------------------------------

    def build_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        cls = _OPTIMIZERS.get(optimizer_name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                f"Choose from {list(_OPTIMIZERS)}."
            )
        if optimizer_name.lower() not in {"adamw", "adamax", "lamb"}:
            kwargs.pop("weight_decay", None) # Only adamw supports weight decay
        return cls(model.parameters(), lr=lr, **kwargs)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        *,
        model: Optional[nn.Module] = None,
        tape: Optional[Any] = None,
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ------------------------------------------------------------------
    # JIT / vmap
    # ------------------------------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        try:
            # PyTorch 2.x
            return torch.compile(func, **kwargs)
        except AttributeError:
            return torch.jit.script(func)

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, Sequence[Optional[int]]] = 0,
        out_axes: int = 0,
    ) -> Callable:
        return torch.vmap(func, in_dims=in_axes, out_dims=out_axes)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def default_device(self) -> str:
        return str(self._device)

    def to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        return tensor.to(device=torch.device(device))

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            self._device = torch.device(kwargs["device"])
        if "enable_tf32" in kwargs and torch.cuda.is_available():
            v = kwargs["enable_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = v
            torch.backends.cudnn.allow_tf32        = v

    def seed(self, value: int) -> None:
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)