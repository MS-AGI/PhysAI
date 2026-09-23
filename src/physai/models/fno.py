"""
physai/models/fno.py

Fourier Neural Operator (FNO) — backend-agnostic implementation.

Architecture (Li et al. 2020, "Fourier Neural Operator for
Parametric Partial Differential Equations"):

    Input  →  Lifting (P)
           →  N × FNO Layers  [ SpectralConv + local W ]
           →  Projection (Q)
           →  Output

Each FNO layer:
    v_{l+1}(x) = σ( K(v_l)(x) + W v_l(x) + b )

    K(v) = F⁻¹[ R(k) · F[v](k) ]     (Fourier integral operator)
    R(k) ∈ ℂ^{d_v × d_v}              (learnable weight tensor, low-mode)
    W    ∈ ℝ^{d_v × d_v}              (local linear transform)

Implementation notes
--------------------
* The SpectralConv is implemented in terms of the backend rfftn / irfftn
  so it runs on PyTorch, JAX, and TensorFlow identically.
* Complex weight tensors R are stored as (real, imag) pairs to avoid
  backend differences in complex gradient handling.
* Supports 1-D, 2-D, and 3-D spatial grids (+ optional time axis).
* Grid input encoding: a grid of normalised coordinates is concatenated
  to the channel dimension before lifting, giving the model positional
  awareness without hardcoded positional encodings.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from physai.backends.base import AbstractBackend, Tensor

# ---------------------------------------------------------------------------
# Complex weight storage
# ---------------------------------------------------------------------------

class ComplexWeight:
    """
    Learnable complex tensor stored as (real, imag) pair.
    Shape: ``(*leading, in_channels, out_channels, *modes)``

    We store separate real/imaginary arrays so that all backends can
    differentiate through them with their real-number autograd systems.
    """

    def __init__(
        self,
        backend: AbstractBackend,
        shape: Tuple[int, ...],
        scale: float = 0.02,
    ) -> None:
        self.backend = backend
        self.shape   = shape

        rng  = np.random.default_rng(seed=None)
        re   = rng.normal(0.0, scale, shape).astype(np.float32)
        im   = rng.normal(0.0, scale, shape).astype(np.float32)

        if backend.name == "torch":
            import torch
            self.re = torch.nn.Parameter(torch.tensor(re))
            self.im = torch.nn.Parameter(torch.tensor(im))
        elif backend.name == "jax":
            import jax.numpy as jnp
            self.re = jnp.array(re)
            self.im = jnp.array(im)
        elif backend.name == "tensorflow":
            import tensorflow as tf
            self.re = tf.Variable(tf.constant(re), trainable=True)
            self.im = tf.Variable(tf.constant(im), trainable=True)
        else:
            if backend.name == "paddle":
                import paddle
                self.re = paddle.create_parameter(shape=shape, dtype="float32")
                self.im = paddle.create_parameter(shape=shape, dtype="float32")
                self.re.set_value(paddle.to_tensor(re))
                self.im.set_value(paddle.to_tensor(im))
            else:
                self.re = backend.tensor(re)
                self.im = backend.tensor(im)

    def as_complex(self) -> Tuple[Tensor, Tensor]:
        """Return (real_part, imag_part) as backend tensors."""
        return self.re, self.im

    def parameters(self) -> List[Tensor]:
        return [self.re, self.im]


# ---------------------------------------------------------------------------
# SpectralConv (backend-agnostic)
# ---------------------------------------------------------------------------

class SpectralConv:
    """
    N-dimensional spectral convolution layer (Fourier integral operator).

    For a spatial field v of shape [B, C_in, *spatial_dims]:
        1. rfft over spatial dims
        2. Multiply low-mode slice by complex weights R
        3. irfft to return to physical space

    Parameters
    ----------
    backend      : AbstractBackend
    in_channels  : C_in
    out_channels : C_out
    modes        : tuple of int — number of Fourier modes kept per spatial dim
    """

    def __init__(
        self,
        backend: AbstractBackend,
        in_channels: int,
        out_channels: int,
        modes: Tuple[int, ...],
    ) -> None:
        self.backend      = backend
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes
        self.ndim         = len(modes)

        # Weight shape: (in_channels, out_channels, *modes)
        w_shape = (in_channels, out_channels) + modes
        self.weight = ComplexWeight(backend, w_shape)

    # ------------------------------------------------------------------
    # Complex multiply (einsum)
    # ------------------------------------------------------------------

    def _complex_mul(
        self,
        x_re: Tensor,
        x_im: Tensor,
        w_re: Tensor,
        w_im: Tensor,
        equation: str,
    ) -> Tuple[Tensor, Tensor]:
        """
        (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        """
        b = self.backend
        out_re = b.einsum(equation, x_re, w_re) - b.einsum(equation, x_im, w_im)
        out_im = b.einsum(equation, x_re, w_im) + b.einsum(equation, x_im, w_re)
        return out_re, out_im

    # ------------------------------------------------------------------
    # 1-D specialisation
    # ------------------------------------------------------------------

    def _forward_1d(self, v: Tensor) -> Tensor:
        """v: [B, C_in, X]"""
        b      = self.backend
        B, C, X = v.shape[0], v.shape[1], v.shape[2]
        m      = self.modes[0]

        # FFT along last axis
        v_ft   = b.rfft(v, axis=-1)          # complex [B, C, X//2+1]

        # Split into (re, im) for backend-agnostic math
        v_re, v_im = _split_complex(b, v_ft)  # each [B, C, X//2+1]

        # Slice low modes
        v_re_low = v_re[..., :m]
        v_im_low = v_im[..., :m]

        w_re, w_im = self.weight.as_complex()   # [C_in, C_out, m]

        # Batch einsum: "b i m, i o m -> b o m"
        out_re, out_im = self._complex_mul(
            v_re_low, v_im_low, w_re, w_im, "bim,iom->bom"
        )

        # Pad back to full spectral size
        pad_re = b.zeros((B, self.out_channels, X // 2 + 1 - m))
        pad_im = b.zeros((B, self.out_channels, X // 2 + 1 - m))
        out_re = b.concatenate([out_re, pad_re], axis=-1)
        out_im = b.concatenate([out_im, pad_im], axis=-1)

        # Reconstruct complex and irfft
        out_ft = _join_complex(b, out_re, out_im)
        return b.irfft(out_ft, n=X, axis=-1)

    # ------------------------------------------------------------------
    # 2-D specialisation
    # ------------------------------------------------------------------

    def _forward_2d(self, v: Tensor) -> Tensor:
        """v: [B, C_in, H, W]"""
        b           = self.backend
        B, C, H, W  = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
        m1, m2      = self.modes

        v_ft        = b.rfft2(v)            # complex [B, C, H, W//2+1]
        v_re, v_im  = _split_complex(b, v_ft)

        # Low-mode slice: first m1 rows, first m2 cols
        v_re_low = v_re[:, :, :m1, :m2]
        v_im_low = v_im[:, :, :m1, :m2]

        w_re, w_im = self.weight.as_complex()   # [C_in, C_out, m1, m2]

        out_re, out_im = self._complex_mul(
            v_re_low, v_im_low, w_re, w_im, "bimn,iomn->bomn"
        )

        # Zero-pad back
        out_re_full = b.zeros((B, self.out_channels, H, W // 2 + 1))
        out_im_full = b.zeros((B, self.out_channels, H, W // 2 + 1))

        # We cannot use in-place assignment cross-backend; build via concat
        zeros_h = b.zeros((B, self.out_channels, H - m1, m2))
        zeros_w_full = b.zeros((B, self.out_channels, H, W // 2 + 1 - m2))

        top_re = b.concatenate([out_re, zeros_h], axis=2)
        top_im = b.concatenate([out_im, zeros_h], axis=2)
        out_re_full = b.concatenate([top_re, zeros_w_full], axis=3)
        out_im_full = b.concatenate([top_im, zeros_w_full], axis=3)

        out_ft = _join_complex(b, out_re_full, out_im_full)
        return b.irfft2(out_ft, s=(H, W))

    # ------------------------------------------------------------------
    # Generic N-D via rfftn
    # ------------------------------------------------------------------

    def _forward_nd(self, v: Tensor) -> Tensor:
        """v: [B, C_in, *spatial_dims]"""
        b      = self.backend
        spatial = list(v.shape[2:])
        axes   = list(range(2, 2 + self.ndim))

        v_ft   = b.rfftn(v, axes=axes)
        v_re, v_im = _split_complex(b, v_ft)

        # Slice modes
        slices = tuple(slice(0, m) for m in self.modes)
        idx    = (slice(None), slice(None)) + slices
        v_re_low = v_re[idx]
        v_im_low = v_im[idx]

        w_re, w_im = self.weight.as_complex()

        # Build einsum equation dynamically
        mode_chars = "mnpqrs"[:self.ndim]
        eq         = f"bi{mode_chars},io{mode_chars}->bo{mode_chars}"
        out_re, out_im = self._complex_mul(
            v_re_low, v_im_low, w_re, w_im, eq
        )

        # Zero-pad
        ft_shape = list(v_ft.shape)
        ft_shape[1] = self.out_channels
        pad_re = b.zeros(tuple(ft_shape))
        pad_im = b.zeros(tuple(ft_shape))

        # Embed low modes into zero-padded array via concatenation chains
        # (safe across all backends — no in-place ops)
        out_re_full = _embed_modes(b, out_re, pad_re, self.modes)
        out_im_full = _embed_modes(b, out_im, pad_im, self.modes)

        s = spatial
        out_ft = _join_complex(b, out_re_full, out_im_full)
        return b.irfftn(out_ft, s=s, axes=axes)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def __call__(self, v: Tensor) -> Tensor:
        if self.ndim == 1:
            return self._forward_1d(v)
        if self.ndim == 2:
            return self._forward_2d(v)
        return self._forward_nd(v)

    def parameters(self) -> List[Tensor]:
        return self.weight.parameters()


# ---------------------------------------------------------------------------
# FNO Layer
# ---------------------------------------------------------------------------

class FNOLayer:
    """
    One FNO residual layer:
        v ← σ( SpectralConv(v) + W·v + b )

    The local linear transform W is implemented as a backend-native
    Dense/Linear layer (1×1 convolution equivalent).
    """

    def __init__(
        self,
        backend: AbstractBackend,
        width: int,
        modes: Tuple[int, ...],
        activation: str = "gelu",
    ) -> None:
        self.backend    = backend
        self.width      = width
        self.modes      = modes
        self.ndim       = len(modes)
        self.activation = activation

        self.spectral   = SpectralConv(backend, width, width, modes)

        # Local linear W: implemented as a [width × width] linear layer
        self._W = backend.build_mlp(
            layer_sizes  = [width, width],
            activation   = "tanh",   # activation not used (single layer)
            use_residual = False,
        )
        # Override: we want a pure linear map. Remove activation from single-layer MLP.
        # For torch: self._W.net is nn.Sequential([Linear(w,w)])
        # This is correct: build_mlp with 2 layers and no activation in last layer.

    def _act(self, x: Tensor) -> Tensor:
        b = self.backend
        _ACTS = {
            "tanh":    b.tanh,
            "relu":    lambda t: 0.5 * (t + b.abs(t)),
            "gelu":    lambda t: t * 0.5 * (1.0 + b.tanh(0.7978845608 * (t + 0.044715 * t ** 3))),
            "silu":    lambda t: t * (1.0 / (1.0 + b.exp(-t))),
        }
        fn = _ACTS.get(self.activation.lower(), b.tanh)
        return fn(x)

    def __call__(self, v: Tensor, jax_params: Any = None) -> Tensor:
        """
        v : [B, width, *spatial_dims]
        """
        b = self.backend

        # Spectral path
        v_spec = self.spectral(v)

        # Local path W·v — treat spatial dims as batch dims via reshape
        shape     = v.shape
        B, C      = shape[0], shape[1]
        spatial   = list(shape[2:])
        n_spatial = int(np.prod(spatial))

        v_flat  = b.reshape(v, (B * n_spatial, C))  # [B*N, C]

        if b.name == "torch":
            w_out = self._W(v_flat)
        elif b.name == "tensorflow":
            w_out = self._W(v_flat, training=False)
        elif b.name == "jax":
            w_out = _jax_apply_mlp(self._W, v_flat, jax_params)
        elif b.name == "paddle":
            w_out = self._W(v_flat)
        else:
            w_out = v_flat

        w_out = b.reshape(w_out, (B, C) + tuple(spatial))

        return self._act(v_spec + w_out)

    def parameters(self) -> List[Tensor]:
        params = self.spectral.parameters()
        params += self.backend.parameters(self._W)
        return params


# ---------------------------------------------------------------------------
# FNO Model
# ---------------------------------------------------------------------------

class FNO:
    """
    Fourier Neural Operator.

    Supports 1-D, 2-D, and 3-D inputs.  The model operates on
    discretised function values on a uniform grid.

    Parameters
    ----------
    backend             : AbstractBackend
    n_input_channels    : number of input channels (e.g. 1 for scalar field
                          + d for grid coords → 1 + d)
    n_output_channels   : number of output channels
    modes               : Fourier modes per spatial dimension
    width               : channel width inside FNO layers
    n_layers            : number of FNO layers
    activation          : activation in FNO layers
    projection_channels : intermediate dim in the output projection MLP
    append_grid         : if True, concatenate normalised grid coordinates
                          to input channels before lifting (default: True)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        n_input_channels: int,
        n_output_channels: int,
        modes: Tuple[int, ...],
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
        projection_channels: int = 128,
        append_grid: bool = True,
    ) -> None:
        self.backend           = backend
        self.n_input_channels  = n_input_channels
        self.n_output_channels = n_output_channels
        self.modes             = modes
        self.width             = width
        self.n_layers          = n_layers
        self.activation        = activation
        self.append_grid       = append_grid
        self.ndim              = len(modes)

        # Actual input dim after grid appending
        in_ch = n_input_channels + (self.ndim if append_grid else 0)

        # Lifting layer P: in_ch → width
        self._lift = backend.build_mlp(
            layer_sizes  = [in_ch, width],
            activation   = activation,
            use_residual = False,
        )

        # FNO layers
        self._layers: List[FNOLayer] = [
            FNOLayer(backend, width, modes, activation)
            for _ in range(n_layers)
        ]

        # Projection Q: width → projection_channels → n_output_channels
        self._proj = backend.build_mlp(
            layer_sizes  = [width, projection_channels, n_output_channels],
            activation   = activation,
            use_residual = False,
        )
        self._jax_params: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Grid encoding
    # ------------------------------------------------------------------

    def _make_grid(self, shape: Tuple[int, ...]) -> Tensor:
        """
        Build a normalised [0, 1]^d grid of shape (1, d, *spatial_dims).
        Concatenated to input channels for positional awareness.
        """
        b   = self.backend
        grids = []
        for size in shape:
            g = b.linspace(0.0, 1.0, size)
            grids.append(g)
        mg = b.meshgrid(*grids, indexing="ij")  # list of [*spatial_dims]
        # Stack and add batch + channel dims
        grid = b.stack(mg, axis=0)              # [d, *spatial_dims]
        grid = b.reshape(grid, (1, self.ndim) + shape)  # [1, d, *spatial_dims]
        return grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, v: Tensor) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor [B, C_in, *spatial_dims]
            Discretised input function on a uniform grid.

        Returns
        -------
        Tensor [B, C_out, *spatial_dims]
        """
        b     = self.backend
        shape = tuple(v.shape[2:])   # spatial dimensions
        B     = v.shape[0]

        # 1. Optionally append grid
        if self.append_grid:
            grid = self._make_grid(shape)
            # Expand grid to batch size
            grid_batch = b.stack([grid[0]] * B, axis=0)  # [B, d, *spatial]
            v = b.concatenate([v, grid_batch], axis=1)

        # 2. Lift: [B, C_in+d, *spatial] → [B, width, *spatial]
        jax_params = self._jax_params if b.name == "jax" else None
        v = self._lift_channels(v, None if jax_params is None else jax_params["lift"])

        # 3. FNO layers
        for i, layer in enumerate(self._layers):
            layer_params = None if jax_params is None else jax_params["layers"][i]
            v = layer(v, layer_params)

        # 4. Project: [B, width, *spatial] → [B, C_out, *spatial]
        proj_params = None if jax_params is None else jax_params["proj"]
        v = self._project_channels(v, proj_params)

        return v

    def _lift_channels(self, v: Tensor, jax_params: Any = None) -> Tensor:
        """Apply lifting MLP channel-wise (shares weights across spatial pts)."""
        return self._apply_channel_mlp(self._lift, v, jax_params)

    def _project_channels(self, v: Tensor, jax_params: Any = None) -> Tensor:
        """Apply projection MLP channel-wise."""
        return self._apply_channel_mlp(self._proj, v, jax_params)

    def _apply_channel_mlp(self, mlp: Any, v: Tensor, jax_params: Any = None) -> Tensor:
        """
        Apply a backend-native MLP to tensor v: [B, C, *spatial]
        by reshaping to [B*N_spatial, C], applying MLP, reshaping back.
        """
        b       = self.backend
        shape   = tuple(v.shape)
        B, C    = shape[0], shape[1]
        spatial = shape[2:]
        N       = int(np.prod(spatial))

        # Channels last for MLP: [B*N, C]
        v_cl   = b.reshape(v, (B, C) + spatial)
        # Move channel dim to last: [B, *spatial, C]
        # Reorder via reshape + transpose equivalent — use einsum trick
        v_flat = b.reshape(
            b.stack(
                [b.reshape(v[:, c], (B, N)) for c in range(C)],
                axis=-1,
            ),
            (B * N, C),
        )

        if b.name == "torch":
            out = mlp(v_flat)
        elif b.name == "tensorflow":
            out = mlp(v_flat, training=False)
        elif b.name == "jax":
            out = _jax_apply_mlp(mlp, v_flat, jax_params)
        elif b.name == "paddle":
            out = mlp(v_flat)
        else:
            out = v_flat

        C_out  = out.shape[-1]
        # Reshape back to [B, *spatial, C_out] → [B, C_out, *spatial]
        out    = b.reshape(out, (B, N, C_out))
        # Swap N and C_out dims: [B, C_out, N] → [B, C_out, *spatial]
        cols   = [b.reshape(out[:, :, c], (B,) + spatial) for c in range(C_out)]
        return b.stack(cols, axis=1)

    # ------------------------------------------------------------------
    # Callable interface (for PDE residuals on continuous inputs)
    # ------------------------------------------------------------------

    @property
    def model_fn(self) -> Callable[[Tensor], Tensor]:
        """
        Returns a continuous interpolation callable u = f(x) for use
        with PDEResidual, by treating each input point as a [1, C, 1]
        grid and evaluating the FNO.

        FNO is inherently a grid-to-grid operator. This compatibility
        adapter gives every point an independent one-cell grid, so there
        is no neighboring grid context and no meaningful spectral mixing.
        Use ``Trainer(grid_inputs=..., grid_targets=...)`` with complete
        fields to train or evaluate the actual FNO operator; use PINN for
        pointwise PDE collocation training.
        """
        def fn(x: Tensor) -> Tensor:
            b   = self.backend
            N   = x.shape[0]
            C   = self.n_input_channels
            # Reshape x to [N, C, 1] — treat each point as a 1-point grid
            v   = b.reshape(x[:, :C], (N, C, 1))
            out = self.forward(v)          # [N, C_out, 1]
            return b.reshape(out, (N, self.n_output_channels))
        return fn

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        if self.backend.name == "jax":
            if self._jax_params is None:
                return []
            import jax
            return list(jax.tree_util.tree_leaves(self._jax_params))
        b      = self.backend
        params = b.parameters(self._lift) + b.parameters(self._proj)
        for layer in self._layers:
            params += layer.parameters()
        return params

    def named_parameters(self) -> dict:
        """Return stable flat parameter names for Trainer checkpoints."""
        return {f"param_{i}": p for i, p in enumerate(self.parameters())}

    def init_jax_params(self, dummy_input: Tensor) -> Any:
        """Initialize and return the full FNO parameter pytree for JAX."""
        if self.backend.name != "jax":
            raise RuntimeError("init_jax_params() is only valid for the JAX backend.")
        import jax
        import jax.numpy as jnp

        def init_mlp(mlp: Any, in_dim: int) -> Any:
            if hasattr(mlp, "init"):
                return mlp.init(mlp.init_key, jnp.zeros((1, in_dim), dtype=dummy_input.dtype))
            return mlp.params

        in_dim = self.n_input_channels + (self.ndim if self.append_grid else 0)
        params = {
            "lift": init_mlp(self._lift, in_dim),
            "layers": [
                {
                    "spectral_re": layer.spectral.weight.re,
                    "spectral_im": layer.spectral.weight.im,
                    "W": init_mlp(layer._W, self.width),
                }
                for layer in self._layers
            ],
            "proj": init_mlp(self._proj, self.width),
        }
        self.set_jax_params(params)
        return params

    def set_jax_params(self, params: Dict[str, Any]) -> None:
        """Bind the supplied pytree to each functional JAX FNO component."""
        if self.backend.name != "jax":
            raise RuntimeError("set_jax_params() is only valid for the JAX backend.")
        self._jax_params = params
        for mlp, mlp_params in [(self._lift, params["lift"]), (self._proj, params["proj"])]:
            if hasattr(mlp, "bound_params"):
                mlp.bound_params = mlp_params
        for layer, layer_params in zip(self._layers, params["layers"]):
            if hasattr(layer._W, "bound_params"):
                layer._W.bound_params = layer_params["W"]
            layer.spectral.weight.re = layer_params["spectral_re"]
            layer.spectral.weight.im = layer_params["spectral_im"]

    def __repr__(self) -> str:
        n = sum(
            int(np.prod(self.backend.to_numpy(
                self.backend.ones(p.shape)
            ).shape))
            for p in self.parameters()
        )
        return (
            f"FNO(backend={self.backend.name}, "
            f"ndim={self.ndim}, modes={self.modes}, "
            f"width={self.width}, layers={self.n_layers}, "
            f"n_params≈{n:,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fno(
    backend: AbstractBackend,
    n_input_channels: int,
    n_output_channels: int,
    modes: Tuple[int, ...],
    width: int = 64,
    n_layers: int = 4,
    activation: str = "gelu",
    projection_channels: int = 128,
    append_grid: bool = True,
) -> FNO:
    """
    Convenience factory for FNO.

    Example — 2-D Navier-Stokes operator:
    >>> fno = build_fno(backend,
    ...     n_input_channels=1, n_output_channels=1,
    ...     modes=(12, 12), width=64, n_layers=4)
    >>> u_next = fno.forward(u_prev)   # [B, 1, H, W]
    """
    return FNO(
        backend             = backend,
        n_input_channels    = n_input_channels,
        n_output_channels   = n_output_channels,
        modes               = modes,
        width               = width,
        n_layers            = n_layers,
        activation          = activation,
        projection_channels = projection_channels,
        append_grid         = append_grid,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jax_apply_mlp(mlp: Any, x: Tensor, params: Any = None) -> Tensor:
    """
    Run a JAX-backend MLP built by ``JAXBackend.build_mlp`` on input ``x``.

    JAX models are functional, so unlike the torch/tf paths (where the
    model object is directly callable) a JAX model needs explicit
    parameters. This mirrors the two shapes ``build_mlp`` can return:

    * ``_FlaxModelHandle`` (preferred path, Flax installed): a mutable
      wrapper around a frozen ``nn.Module`` exposing ``.init``/``.apply``
      and a ``bound_params`` slot. Parameters are lazily initialised on
      first call and cached on the handle so every subsequent forward
      pass reuses the same weights instead of re-randomising them.
    * ``_PureJAXMLP`` (no-Flax fallback): a NamedTuple already carrying
      initialised ``.params`` plus an ``.apply(params, x)`` function.

    Without this, the JAX path of FNO silently skipped lifting,
    projection, and every FNOLayer's local linear transform W — it just
    returned the input unchanged, training on a broken architecture with
    no error raised.
    """
    if hasattr(mlp, "bound_params"):  # _FlaxModelHandle
        if params is None:
            params = mlp.bound_params
        if params is None:
            mlp.bound_params = mlp.init(mlp.init_key, x)
            params = mlp.bound_params
        return mlp.apply(params, x)
    # _PureJAXMLP: params already initialised inside build_mlp
    return mlp.apply(mlp.params if params is None else params, x)


def _split_complex(backend: AbstractBackend, z: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Split a complex tensor into (real, imag) using backend-native ops.
    Works for PyTorch (z.real / z.imag), TF (tf.math.real/imag),
    and JAX (z.real / z.imag).
    """
    b = backend
    if b.name == "torch":
        return z.real, z.imag
    if b.name == "jax":
        import jax.numpy as jnp
        return jnp.real(z), jnp.imag(z)
    if b.name == "tensorflow":
        import tensorflow as tf
        return tf.math.real(z), tf.math.imag(z)
    if b.name == "paddle":
        import paddle
        return paddle.real(z), paddle.imag(z)
    # Fallback: assume numpy-like
    import numpy as np
    return np.real(z), np.imag(z)


def _join_complex(
    backend: AbstractBackend,
    re: Tensor,
    im: Tensor,
) -> Tensor:
    """
    Construct a complex tensor from real and imaginary parts.
    """
    b = backend
    if b.name == "torch":
        return re + 1j * im
    if b.name == "jax":
        import jax.numpy as jnp
        return re + 1j * im
    if b.name == "tensorflow":
        import tensorflow as tf
        return tf.complex(re, im)
    if b.name == "paddle":
        import paddle
        return paddle.complex(re, im)
    import numpy as np
    return re + 1j * im


def _embed_modes(
    backend: AbstractBackend,
    low: Tensor,
    full: Tensor,
    modes: Tuple[int, ...],
) -> Tensor:
    """
    Embed low-mode tensor into a zero-padded full-spectrum tensor
    by concatenating zeros along each mode axis.  Avoids in-place ops.

    ``low``  : [..., m1, m2, …]
    ``full`` : [..., M1, M2, …]  (target shape)
    """
    b = backend
    result = low
    for axis_offset, (m, M) in enumerate(zip(modes, full.shape[2:])):
        ax      = 2 + axis_offset
        shape   = list(result.shape)
        shape[ax] = M - m
        zeros   = b.zeros(tuple(shape))
        result  = b.concatenate([result, zeros], axis=ax)
    return result


__all__ = [
    "FNO",
    "build_fno",
    "FNOLayer",
    "SpectralConv",
    "ComplexWeight",
]
