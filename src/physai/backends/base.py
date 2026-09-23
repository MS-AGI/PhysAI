"""
physai/backends/base.py

Abstract execution context interface for PhysAI multi-backend support.
Defines the contract that all backend implementations must satisfy.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore
# ---------------------------------------------------------------------------
# Type aliases (backend-agnostic)
# ---------------------------------------------------------------------------
Tensor = Any          # concrete type depends on the active backend
DType  = Any
Shape  = Tuple[int, ...]


class BackendCapabilities:
    """Declarative capability flags reported by each backend."""

    def __init__(
        self,
        *,
        supports_jit: bool = False,
        supports_vmap: bool = False,
        supports_complex: bool = False,
        supports_sparse: bool = False,
        supports_mixed_precision: bool = False,
        native_fft: bool = True,
    ) -> None:
        self.supports_jit              = supports_jit
        self.supports_vmap             = supports_vmap
        self.supports_complex          = supports_complex
        self.supports_sparse           = supports_sparse
        self.supports_mixed_precision  = supports_mixed_precision
        self.native_fft                = native_fft

    def __repr__(self) -> str:  # pragma: no cover
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"BackendCapabilities({attrs})"


class AbstractBackend(abc.ABC):
    """
    Abstract base class for all PhysAI execution backends.

    Subclasses must implement every ``@abc.abstractmethod`` below.
    Optional capabilities (JIT, vmap, …) should raise ``NotImplementedError``
    with a descriptive message when not supported.

    Lifecycle
    ---------
    1.  Instantiate the backend (``__init__`` may set device, dtype defaults).
    2.  Optionally call ``configure(**kwargs)`` for runtime tuning.
    3.  Build models via the framework-specific ``nn`` attribute or helpers.
    4.  Call ``grad``, ``value_and_grad``, etc. during training.
    """

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'torch', 'jax', 'tensorflow'."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Static capability descriptor for this backend."""

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
    ) -> Tensor:
        """Convert ``data`` (list, numpy array, …) to a backend tensor."""

    @abc.abstractmethod
    def zeros(self, shape: Shape, dtype: Optional[DType] = None) -> Tensor:
        """Return a zero-filled tensor of ``shape``."""

    @abc.abstractmethod
    def ones(self, shape: Shape, dtype: Optional[DType] = None) -> Tensor:
        """Return a one-filled tensor of ``shape``."""

    @abc.abstractmethod
    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: Optional[DType] = None,
    ) -> Tensor:
        """Evenly-spaced values in ``[start, stop]``."""

    @abc.abstractmethod
    def meshgrid(
        self,
        *tensors: Tensor,
        indexing: str = "ij",
    ) -> List[Tensor]:
        """N-dimensional mesh grid from 1-D coordinate tensors."""

    @abc.abstractmethod
    def stack(self, tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
        """Stack a sequence of tensors along a new ``axis``."""

    @abc.abstractmethod
    def concatenate(self, tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
        """Concatenate tensors along an existing ``axis``."""

    @abc.abstractmethod
    def reshape(self, tensor: Tensor, shape: Shape) -> Tensor:
        """Return a reshaped view (or copy) of ``tensor``."""

    @abc.abstractmethod
    def cast(self, tensor: Tensor, dtype: DType) -> Tensor:
        """Cast ``tensor`` to ``dtype``."""

    @abc.abstractmethod
    def to_numpy(self, tensor: Tensor) -> "np.ndarray":  # noqa: F821
        """Convert a backend tensor to a NumPy array."""

    # ------------------------------------------------------------------
    # Mathematical primitives
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def mean(self, tensor: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
        """Reduce mean, optionally along ``axis``."""

    @abc.abstractmethod
    def sum(self, tensor: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
        """Reduce sum, optionally along ``axis``."""

    @abc.abstractmethod
    def abs(self, tensor: Tensor) -> Tensor:
        """Element-wise absolute value."""

    @abc.abstractmethod
    def sqrt(self, tensor: Tensor) -> Tensor:
        """Element-wise square root."""

    @abc.abstractmethod
    def square(self, tensor: Tensor) -> Tensor:
        """Element-wise square."""

    @abc.abstractmethod
    def exp(self, tensor: Tensor) -> Tensor:
        """Element-wise exponential."""

    @abc.abstractmethod
    def log(self, tensor: Tensor) -> Tensor:
        """Element-wise natural logarithm."""

    @abc.abstractmethod
    def sin(self, tensor: Tensor) -> Tensor:
        """Element-wise sine."""

    @abc.abstractmethod
    def cos(self, tensor: Tensor) -> Tensor:
        """Element-wise cosine."""

    @abc.abstractmethod
    def atan(self, tensor: Tensor) -> Tensor:
        """Element-wise arctangent (inverse tangent)."""

    @abc.abstractmethod
    def tanh(self, tensor: Tensor) -> Tensor:
        """Element-wise hyperbolic tangent."""

    @abc.abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication."""

    @abc.abstractmethod
    def einsum(self, equation: str, *operands: Tensor) -> Tensor:
        """Einstein summation."""

    # ------------------------------------------------------------------
    # FFT operations (required — every backend must provide these)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def rfft(self, tensor: Tensor, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """Real-input 1-D FFT → complex half-spectrum."""

    @abc.abstractmethod
    def irfft(self, tensor: Tensor, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """Inverse of ``rfft``."""

    @abc.abstractmethod
    def rfft2(self, tensor: Tensor, s: Optional[Tuple[int, int]] = None) -> Tensor:
        """Real-input 2-D FFT over the last two axes."""

    @abc.abstractmethod
    def irfft2(self, tensor: Tensor, s: Optional[Tuple[int, int]] = None) -> Tensor:
        """Inverse of ``rfft2``."""

    @abc.abstractmethod
    def rfftn(
        self,
        tensor: Tensor,
        s: Optional[Sequence[int]] = None,
        axes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Real-input N-D FFT."""

    @abc.abstractmethod
    def irfftn(
        self,
        tensor: Tensor,
        s: Optional[Sequence[int]] = None,
        axes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Inverse of ``rfftn``."""

    # ------------------------------------------------------------------
    # Automatic differentiation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def grad(
        self,
        func: Callable[..., Tensor],
        argnums: Union[int, Sequence[int]] = 0,
        *,
        mode: str = "reverse",
    ) -> Callable[..., Tensor]:
        """
        Return a function that computes the gradient of ``func`` w.r.t.
        the arguments at positions ``argnums``.

        For eager backends (PyTorch, TensorFlow) this wraps the call in
        the appropriate tape/autograd context.

        Parameters
        ----------
        mode : which differentiation strategy to use.
            - ``"reverse"`` (default): standard vector-Jacobian-product
              (backprop) AD. Cheapest when the function has few outputs
              relative to inputs — the common PINN case (scalar/small
              vector field, many collocation points).
            - ``"forward"``: jacobian-vector-product AD, built by sweeping
              a one-hot tangent over each input dimension. Cheaper than
              reverse mode when the function has many outputs relative to
              inputs (e.g. per-point Jacobians of vector fields, or a
              small number of spatial dims with many output components as
              in coupled/mixed residual systems).
            - ``"taylor"``: forward-over-forward AD (nested JVP), used to
              obtain second-order directional derivatives (e.g. diagonal
              Hessian terms for Laplacians) in a single forward sweep
              without materializing a full reverse-mode graph twice.
        """

    @abc.abstractmethod
    def jvp(
        self,
        func: Callable[..., Tensor],
        inputs: Tensor,
        tangents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward-mode Jacobian-vector product: returns ``(value, J @ tangents)``
        for ``func`` evaluated at ``inputs`` with direction ``tangents``.
        Backends without native forward-mode AD may emulate this via
        double-backward (reverse-over-reverse) but should document the cost.
        """

    @abc.abstractmethod
    def value_and_grad(
        self,
        func: Callable[..., Tensor],
        argnums: Union[int, Sequence[int]] = 0,
    ) -> Callable[..., Tuple[Tensor, Tensor]]:
        """Return ``(value, gradient)`` together to avoid redundant passes."""

    @abc.abstractmethod
    def jacobian(
        self,
        func: Callable[..., Tensor],
        inputs: Tensor,
    ) -> Tensor:
        """Full Jacobian matrix of ``func`` evaluated at ``inputs``."""

    @abc.abstractmethod
    def hessian(
        self,
        func: Callable[..., Tensor],
        inputs: Tensor,
    ) -> Tensor:
        """Full Hessian matrix of ``func`` evaluated at ``inputs``."""

    # ------------------------------------------------------------------
    # Neural-network model interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_mlp(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        *,
        use_residual: bool = False,
        dtype: Optional[DType] = None,
    ) -> Any:
        """
        Construct a fully-connected MLP with the given ``layer_sizes``.
        Returns a framework-native model object (``nn.Module``, ``keras.Model``, …).
        """

    @abc.abstractmethod
    def parameters(self, model: Any) -> List[Tensor]:
        """Return all trainable parameters of ``model`` as a flat list."""

    @abc.abstractmethod
    def zero_grad(self, model: Any) -> None:
        """Zero accumulated gradients (no-op for tape-based backends)."""

    # ------------------------------------------------------------------
    # Optimiser helpers
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_optimizer(
        self,
        model: Any,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> Any:
        """Return a framework-native optimiser bound to ``model``'s parameters."""

    @abc.abstractmethod
    def optimizer_step(
        self,
        optimizer: Any,
        loss: Tensor,
        *,
        model: Optional[Any] = None,
        tape: Optional[Any] = None,
    ) -> None:
        """
        Perform one gradient-descent step.

        Parameters
        ----------
        optimizer : framework-native optimiser
        loss      : scalar loss tensor
        model     : required for TensorFlow (to access ``trainable_variables``)
        tape      : required for TensorFlow (``GradientTape``)
        """

    # ------------------------------------------------------------------
    # JIT / vmap (optional — raise NotImplementedError if unsupported)
    # ------------------------------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        """
        Return a JIT-compiled version of ``func``.
        Default: identity (no compilation).
        """
        return func  # safe fallback

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, Sequence[Optional[int]]] = 0,
        out_axes: int = 0,
    ) -> Callable:
        """
        Vectorise ``func`` over a batch axis.
        Default: raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not support vmap."
        )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def default_device(self) -> str:
        """Return a string description of the default compute device."""

    @abc.abstractmethod
    def to_device(self, tensor: Tensor, device: str) -> Tensor:
        """Move ``tensor`` to ``device``."""

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        """
        Optional runtime configuration hook.
        Subclasses may override to accept backend-specific settings
        (e.g., ``enable_tf32``, ``x64``, ``memory_growth``).
        """

    def seed(self, value: int) -> None:
        """Set the global random seed for reproducibility."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PhysAI Backend: {self.name} | device={self.default_device()}>"