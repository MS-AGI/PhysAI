"""
physai/models/pinn.py

Physics-Informed Neural Network (PINN) model wrapper.

Wraps a backend-native MLP with:
  * Fourier feature embedding (Random Fourier Features / positional encoding)
  * Hard constraint enforcement via domain transformation
  * Per-output scaling for multi-physics problems
  * Gradient checkpointing hooks (PyTorch)
  * A clean forward interface: model_fn(x) → u

Design
------
The PINN is deliberately separated from training logic. It exposes only:
    pinn.forward(x)          → Tensor
    pinn.model_fn            → Callable[[Tensor], Tensor]  (closure)
    pinn.named_parameters()  → backend-specific param list

so that PDE residuals and loss functions only ever receive a callable.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from physai.backends.base import AbstractBackend, Tensor

# ---------------------------------------------------------------------------
# Fourier Feature Embedding
# ---------------------------------------------------------------------------

class FourierEmbedding:
    """
    Random Fourier Features embedding:
        φ(x) = [sin(Bx), cos(Bx)]   where B ~ N(0, σ²)

    Maps low-dimensional inputs to a high-dimensional feature space that
    mitigates spectral bias (the tendency of deep networks to learn low
    frequencies first).

    References
    ----------
    Tancik et al. (2020) "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains."
    """

    def __init__(
        self,
        backend: AbstractBackend,
        n_input: int,
        n_frequencies: int = 128,
        sigma: float = 1.0,
        trainable: bool = False,
    ) -> None:
        self.backend       = backend
        self.n_input       = n_input
        self.n_frequencies = n_frequencies
        self.sigma         = sigma
        self.trainable     = trainable
        self._B: Optional[Tensor] = None
        self._init_B()

    def _init_B(self) -> None:
        import numpy as np
        rng = np.random.default_rng(seed=42)
        B_np = rng.normal(0.0, self.sigma, (self.n_input, self.n_frequencies)).astype("float32")
        self._B = self.backend.tensor(B_np)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_frequencies

    def __call__(self, x: Tensor) -> Tensor:
        b  = self.backend
        # x: [N, n_input]  →  xB: [N, n_frequencies]
        xB = b.matmul(x, self._B)  # [N, F]
        return b.concatenate(
            [b.sin(2.0 * math.pi * xB), b.cos(2.0 * math.pi * xB)],
            axis=-1,
        )


# ---------------------------------------------------------------------------
# Hard constraint wrappers
# ---------------------------------------------------------------------------

class DirichletConstraint:
    """
    Enforces u = g on the boundary exactly by multiplying by a distance
    function D(x) that is zero on ∂Ω:

        ũ(x) = D(x) * N(x) + g(x)

    where N(x) is the unconstrained network output.

    Parameters
    ----------
    distance_fn  : D(x) → Tensor  — smooth function, zero on boundary
    boundary_fn  : g(x) → Tensor  — prescribed Dirichlet values
    """

    def __init__(
        self,
        distance_fn: Callable[[Tensor], Tensor],
        boundary_fn: Callable[[Tensor], Tensor],
    ) -> None:
        self.distance_fn  = distance_fn
        self.boundary_fn  = boundary_fn

    def apply(self, x: Tensor, raw_output: Tensor) -> Tensor:
        D = self.distance_fn(x)[..., None]   # broadcast over output channels
        g = self.boundary_fn(x)
        return D * raw_output + g


# ---------------------------------------------------------------------------
# Output scaler
# ---------------------------------------------------------------------------

class OutputScaler:
    """
    Per-channel affine rescaling of model outputs.
    Helps when different physical fields have vastly different magnitudes.

    ũ_i = scale_i * u_i + shift_i
    """

    def __init__(
        self,
        backend: AbstractBackend,
        scales: Sequence[float],
        shifts: Optional[Sequence[float]] = None,
    ) -> None:
        self.backend = backend
        self._scales = backend.tensor(list(scales))
        self._shifts = backend.tensor(
            list(shifts) if shifts else [0.0] * len(scales)
        )

    def apply(self, output: Tensor) -> Tensor:
        return output * self._scales + self._shifts


# ---------------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------------

class PINN:
    """
    Physics-Informed Neural Network.

    Parameters
    ----------
    backend        : AbstractBackend
    layer_sizes    : (n_in, h1, h2, …, n_out)  — after embedding if used
    activation     : activation function name
    use_residual   : enable residual connections in the MLP
    use_fourier    : prepend a Fourier feature embedding
    fourier_freqs  : number of Fourier frequencies per dimension
    fourier_sigma  : scale of Fourier frequency matrix
    constraint     : optional DirichletConstraint
    output_scaler  : optional OutputScaler
    dtype          : override backend default dtype
    """

    def __init__(
        self,
        backend: AbstractBackend,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        *,
        use_residual: bool = False,
        use_fourier:  bool = False,
        fourier_freqs: int = 128,
        fourier_sigma: float = 1.0,
        constraint: Optional[DirichletConstraint] = None,
        output_scaler: Optional[OutputScaler] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        self.backend       = backend
        self.constraint    = constraint
        self.output_scaler = output_scaler

        # Optional Fourier embedding
        self.embedding: Optional[FourierEmbedding] = None
        if use_fourier:
            n_in = layer_sizes[0]
            self.embedding = FourierEmbedding(
                backend,
                n_input       = n_in,
                n_frequencies = fourier_freqs,
                sigma         = fourier_sigma,
            )
            # Replace first layer size with embedding output dim
            layer_sizes = (self.embedding.output_dim,) + tuple(layer_sizes[1:])

        # Build the core MLP via backend
        self._model = backend.build_mlp(
            layer_sizes  = layer_sizes,
            activation   = activation,
            use_residual = use_residual,
            dtype        = dtype,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate the PINN at input coordinates ``x``.

        Parameters
        ----------
        x : Tensor of shape [N, n_in]

        Returns
        -------
        Tensor of shape [N, n_out]
        """
        b = self.backend

        # 1. Fourier embedding (optional)
        h = self.embedding(x) if self.embedding is not None else x

        # 2. MLP core — framework-native forward
        if b.name == "torch":
            raw = self._model(h)
        elif b.name == "jax":
            # Flax model: requires (params, x); we store params externally
            # For pure-JAX fallback, call apply directly
            if hasattr(self._model, "apply"):
                raw = self._model.apply(self._init_params, h)
            else:
                # _PureJAXMLP
                raw = self._model.apply(self._model.params, h)
        elif b.name == "tensorflow":
            raw = self._model(h, training=False)
        else:
            raw = self._model(h)

        # 3. Hard constraint (optional)
        if self.constraint is not None:
            raw = self.constraint.apply(x, raw)

        # 4. Output scaling (optional)
        if self.output_scaler is not None:
            raw = self.output_scaler.apply(raw)

        return raw

    # ------------------------------------------------------------------
    # Convenient callable interface for PDE residuals
    # ------------------------------------------------------------------

    @property
    def model_fn(self) -> Callable[[Tensor], Tensor]:
        """
        Returns a pure function  u = f(x)  suitable for passing to
        PDEResidual.__call__ and backend.grad(…).

        The closure captures ``self`` but is otherwise stateless from the
        caller's perspective.
        """
        return self.forward

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        return self.backend.parameters(self._model)

    def named_parameters(self) -> Dict[str, Tensor]:
        b = self.backend
        if b.name == "torch":
            return dict(self._model.named_parameters())
        return {f"param_{i}": p for i, p in enumerate(self.parameters())}

    def zero_grad(self) -> None:
        self.backend.zero_grad(self._model)

    # ------------------------------------------------------------------
    # JAX parameter initialisation helper
    # ------------------------------------------------------------------

    def init_jax_params(self, dummy_input: Tensor) -> Any:
        """
        For Flax models: initialise parameters and store them internally.
        Call this before the first forward pass in JAX mode.
        """
        import jax
        if not hasattr(self._model, "init"):
            return  # pure-JAX MLP already has params
        # If a Fourier embedding is enabled, forward() feeds the network
        # the *embedded* features (a different, larger dimension), not
        # the raw input. Params must be initialised against that same
        # shape or every real forward() call hits a matmul mismatch.
        h = self.embedding(dummy_input) if self.embedding is not None else dummy_input
        key = getattr(self._model, "init_key", jax.random.PRNGKey(0))
        params = self._model.init(key, h)
        self.set_jax_params(params)
        return self._init_params

    def set_jax_params(self, params: Any) -> None:
        """
        Rebind the Flax parameter pytree used by ``forward()``.

        This MUST be the only way JAX training code updates parameters —
        forward() reads ``self._init_params``, so writing to any other
        attribute (e.g. an external trainer's own copy of the params) has
        no effect on the computation graph and silently yields zero
        gradients from ``jax.grad``.
        """
        self._init_params = params
        if hasattr(self._model, "init"):
            # Keep AbstractBackend.parameters()/named_parameters()/save()
            # in sync too (they read `bound_params` off the mutable
            # _FlaxModelHandle wrapper, since the underlying Flax Module
            # itself is a frozen dataclass and can't hold this state).
            self._model.bound_params = params

    # ------------------------------------------------------------------
    # Save / load (backend-agnostic via numpy)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import numpy as np
        params_np = {
            k: self.backend.to_numpy(v)
            for k, v in self.named_parameters().items()
        }
        np.savez(path, **params_np)

    def __repr__(self) -> str:
        n_params = sum(
            int(self.backend.to_numpy(
                self.backend.sum(self.backend.ones(p.shape))
            ))
            for p in self.parameters()
        )
        return (
            f"PINN(backend={self.backend.name}, "
            f"embedding={self.embedding is not None}, "
            f"n_params={n_params:,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pinn(
    backend: AbstractBackend,
    n_input: int,
    n_output: int,
    hidden_sizes: Sequence[int] = (128, 128, 128, 128),
    activation: str = "tanh",
    *,
    use_residual: bool = False,
    use_fourier: bool = False,
    fourier_freqs: int = 128,
    fourier_sigma: float = 1.0,
    constraint: Optional[DirichletConstraint] = None,
    output_scaler: Optional[OutputScaler] = None,
    dtype: Optional[Any] = None,
) -> PINN:
    """
    Convenience factory for building a PINN from flat hyperparameters.

    Example
    -------
    >>> pinn = build_pinn(backend, n_input=2, n_output=1,
    ...                   hidden_sizes=(128,)*4, use_fourier=True)
    >>> u = pinn.model_fn(x)
    """
    layer_sizes = (n_input,) + tuple(hidden_sizes) + (n_output,)
    return PINN(
        backend        = backend,
        layer_sizes    = layer_sizes,
        activation     = activation,
        use_residual   = use_residual,
        use_fourier    = use_fourier,
        fourier_freqs  = fourier_freqs,
        fourier_sigma  = fourier_sigma,
        constraint     = constraint,
        output_scaler  = output_scaler,
        dtype          = dtype,
    )


__all__ = [
    "PINN",
    "build_pinn",
    "FourierEmbedding",
    "DirichletConstraint",
    "OutputScaler",
]