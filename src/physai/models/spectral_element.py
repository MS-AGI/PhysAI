"""
physai/models/spectral_element.py

Unified Spectral Element Neural Operator (USENO), mapped into CP
(Canonical Polyadic) rank space, following the reference architecture in
Feugmo & Pankaczy, "Neural Spectral Element Methods for stiff multiphysics
PDEs" (arXiv:2606.02335).

This module contains the CP math and model implementation. Numerical solver
adapters live in ``physai.solvers.solver``; the associated training helper
and builder live in ``physai.models.spectralpinn``.

Backend support
---------------
This module is backend-agnostic across every ``AbstractBackend``
(``torch``, ``jax``, ``tensorflow``, ``paddle``): the per-step math in
``USENOCPModule.get_cp_factors`` / ``pseudospectral_1d_product`` /
``compute_residuals_and_boundaries`` is expressed entirely through
``self.backend`` primitives (``matmul``, ``stack``, ``reshape``, ...), and
the one-time setup math (Chebyshev nodes/Vandermonde/derivative matrices,
``build_separable_linear_operator``) is now plain NumPy — these constants
never need gradients and are converted to the active backend's tensor type
once, via ``backend.tensor(...)``, at construction time.

Two exceptions, both intentional and backend-checked at the call site
rather than hidden:

* ``USENOCPModule`` subclasses ``torch.nn.Module`` when built on the Torch
  backend; for other backends it is a plain object. This module attempts to
  import PyTorch at import time to define that subclass, but PyTorch is not
  required when a non-Torch backend is used.
* ``DynamicLossBalancer`` (the gradient-norm loss-weighting scheme) uses
  ``torch.autograd.grad`` directly and is, by design, only ever
  instantiated for the torch backend — see ``USENOPINNTrainer.__init__``
  in ``spectralpinn.py``, which sets ``self.balancer = None`` on every
  other backend. Gradient-norm-based loss balancing needs a real
  per-parameter gradient decomposition, which the reference paper only
  specifies via reverse-mode autodiff; jax/tf/paddle keep the three loss
  terms at fixed unit weight instead of getting a silently-wrong port of
  this.

The implementation also addresses two numerical issues:

1. **Basis mismatch (the serious one).** The module's boundary evaluation
   vectors are ``T_i(-1) = (-1)^i`` and ``T_i(1) = 1`` — i.e. they assume a
   *Chebyshev* polynomial basis. But the previous
   ``pseudospectral_1d_product`` reconstructed physical-space values with
   ``torch.fft.irfft``/``rfft``, which is only a valid analysis/synthesis
   pair for a *Fourier* (trigonometric) basis. Multiplying two
   Chebyshev-coefficient vectors via a Fourier IFFT/FFT round-trip computes
   nothing physically meaningful — it silently mixes two incompatible
   spectral representations. This module now uses an explicit Chebyshev
   synthesis matrix (Vandermonde evaluation of ``T_i`` at
   Chebyshev-Gauss-Lobatto nodes) to go coefficients -> physical space, and
   a least-squares Chebyshev analysis matrix (its pseudo-inverse) to go
   back, matching the boundary vectors' basis exactly.

2. **Missing flux (C1) continuity.** The reference paper is explicit that
   "value continuity is exact by construction; the diffusive flux is
   enforced by a C1 penalty ... essential for stiff transport problems."
   The previous ``continuity_loss`` only enforced C0 (value) continuity
   between adjacent elements. ``USENOCPModule`` now also emits derivative
   (flux) boundary evaluations — using the exact analytic Chebyshev
   derivative at the endpoints, ``T_i'(-1) = (-1)^(i+1) i^2`` and
   ``T_i'(1) = i^2`` — and ``USENOPINNTrainer`` penalizes their mismatch
   across element interfaces as a third loss term.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Dict, Tuple, Callable, Optional
from physai.backends import AbstractBackend, get_backend

# torch is an OPTIONAL dependency of this module now: it's only needed
# when a torch backend is actually in play (USENOCPModule subclassing
# nn.Module for torch, DynamicLossBalancer's autograd-based weighting).
# jax/tensorflow/paddle users can import and use everything else in this
# file — including USENOCPModule itself — without torch installed.
try:
    import torch as _torch
    import torch.nn as _nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised in torch-less/broken-torch envs
    # Broad except (not just ImportError): a torch package can be present
    # but fail to actually load (e.g. missing CUDA shared libs in a
    # CPU-only container) — either way, this module must still work for
    # jax/tensorflow/paddle users, so any failure here just means "treat
    # torch as unavailable" rather than crashing the whole import.
    _torch = None
    _nn = None
    _TORCH_AVAILABLE = False

# The concrete array type depends on the active backend. Keep annotations
# backend-neutral rather than referring to ``torch.Tensor`` in a module
# where PyTorch is optional.
BackendTensor = Any


def _to_numpy_host(value: Any) -> np.ndarray:
    """
    Best-effort conversion of an arbitrary array-like — a plain list, a
    NumPy array, or any backend's tensor type (torch, jax, tensorflow,
    paddle) — to a host-side NumPy array, without importing any of those
    frameworks unless ``value`` actually is one of their tensor types.
    Used only for the one-time, non-differentiable setup constants this
    module builds (e.g. ``element_widths``), never on the hot per-step
    path.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple, int, float)):
        return np.asarray(value)
    # torch.Tensor / paddle.Tensor both expose .detach()/.cpu()/.numpy().
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    # jax.Array and anything else numpy already knows how to coerce.
    return np.asarray(value)


# ---------------------------------------------------------------------------
# CP-tensor linear algebra
# ---------------------------------------------------------------------------

class TensorCPMath:
    """
    Core linear algebra utility class for handling CP (Canonical Polyadic)
    tensors. Computes high-dimensional inner products and norms using
    purely 1D operations to bypass the curse of dimensionality.
    """
    @staticmethod
    def compute_cp_inner_product(
        factors_A: List[BackendTensor], factors_B: List[BackendTensor], backend=None,
    ) -> BackendTensor:
        """
        Computes the inner product between two CP tensors A and B across batches.
        Formula: <A, B> = Sum_{r1, r2} Prod_d <factors_A[d][:, r1], factors_B[d][:, r2]>
        Shapes: factors_A[d] -> (Batch, Rank_A, N_modes)
                factors_B[d] -> (Batch, Rank_B, N_modes)
        Returns: (Batch,) tensor of inner products.

        Pass ``backend`` (any ``AbstractBackend``) to use the
        backend-agnostic path below — this is what every caller in this
        codebase does (``USENOPINNTrainer._losses``). The
        ``backend=None`` branch is a legacy torch/NumPy-duck-typed
        fallback for direct, torch-tensor callers outside this package;
        it is not exercised anywhere in this codebase and is kept only
        for backward compatibility with external callers written against
        the pre-multi-backend API.
        """
        batch_size = factors_A[0].shape[0]
        rank_A = factors_A[0].shape[1]
        rank_B = factors_B[0].shape[1]
        dims = len(factors_A)

        # Initialize the interaction matrix with ones: shape (Batch, Rank_A, Rank_B)
        if backend is not None:
            interaction = backend.ones((batch_size, rank_A, rank_B), dtype=factors_A[0].dtype)
            for a, b in zip(factors_A, factors_B):
                interaction = interaction * backend.einsum("bri,bsi->brs", a, b)
            return backend.sum(interaction, axis=(1, 2))

        interaction = factors_A[0][:, :1, :1] * 0 + 1

        for d in range(dims):
            # Handle complex inner products via conjugation.
            is_complex = getattr(factors_B[d], "is_complex", None)
            complex_value = is_complex() if callable(is_complex) else False
            b_matrix = factors_B[d].conj().transpose(1, 2) if complex_value else factors_B[d].transpose(1, 2)

            # (Batch, Rank_A, N_modes) x (Batch, N_modes, Rank_B) -> (Batch, Rank_A, Rank_B)
            dim_inner = factors_A[d] @ b_matrix
            interaction = interaction * dim_inner

        # Sum over all rank combinations to get the final scalar inner product per batch item
        result = interaction.sum(dim=(1, 2))

        # If the result is still carrying a 0j complex tail due to precision, cast to real
        is_complex = getattr(result, "is_complex", None)
        if callable(is_complex) and is_complex():
            return result.real
        return result

    @staticmethod
    def compute_cp_norm_squared(
        factors: List[BackendTensor], backend=None,
    ) -> BackendTensor:
        """Computes the squared L2 norm of a CP tensor: <factors, factors>"""
        return TensorCPMath.compute_cp_inner_product(factors, factors, backend=backend)


# ---------------------------------------------------------------------------
# Dynamic multi-objective loss balancing
# ---------------------------------------------------------------------------

class DynamicLossBalancer:
    """
    Industrial gradient-matching system (Modified Neural Tangent Kernel alignment).
    Dynamically balances weights of multi-objective losses to prevent gradient pathologies.
    """
    def __init__(self, num_losses: int, alpha: float = 0.9):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "DynamicLossBalancer requires PyTorch (torch.autograd.grad-based "
                "gradient-norm weighting) — it is only ever instantiated for the "
                "torch backend; see USENOPINNTrainer.__init__ in spectralpinn.py."
            )
        torch = _torch
        self.num_losses = num_losses
        self.alpha = alpha
        self.weights = torch.ones(num_losses, dtype=torch.float32)

    def update_weights(
        self, losses: List[BackendTensor], shared_parameters: List[BackendTensor],
    ):
        """Adjusts objective loss weights based on the standard deviation of their gradients."""
        torch = _torch
        device = losses[0].device
        self.weights = self.weights.to(device)
        grads = []

        for i, loss in enumerate(losses):
            # HARDENED GUARD: Force conversion to a real float scalar right before autograd
            if loss.is_complex():
                real_loss = torch.abs(loss)
            else:
                real_loss = loss

            grad_norms = []
            grad_list = torch.autograd.grad(real_loss, shared_parameters, retain_graph=True, allow_unused=True)
            for g in grad_list:
                if g is not None:
                    grad_norms.append(g.norm(2))
            if len(grad_norms) > 0:
                grads.append(torch.stack(grad_norms).mean())
            else:
                grads.append(torch.tensor(1.0, device=device))

        grad_stack = torch.stack(grads)
        mean_grad = grad_stack.mean()

        # Compute targeted adaptive weights
        target_weights = mean_grad / (grad_stack + 1e-8)

        # Apply exponential moving average updating for training path smoothness
        self.weights = self.alpha * self.weights + (1.0 - self.alpha) * target_weights
        # Normalize weights to preserve global scale
        self.weights = self.weights / (self.weights.sum() + 1e-8) * self.num_losses


# ---------------------------------------------------------------------------
# Chebyshev analysis / synthesis helpers
# ---------------------------------------------------------------------------

def _chebyshev_gauss_lobatto_nodes(m: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1], x_j = cos(pi * j / (m - 1))."""
    j = np.arange(m, dtype=np.float64)
    return np.cos(np.pi * j / (m - 1))


def _chebyshev_vandermonde(x: np.ndarray, n_modes: int) -> np.ndarray:
    """
    Evaluate T_0..T_{n_modes-1} at points ``x`` via the standard three-term
    recurrence T_0=1, T_1=x, T_{k+1}=2x*T_k - T_{k-1}.

    Returns a (len(x), n_modes) matrix V with V[j, i] = T_i(x_j) — the exact
    Chebyshev synthesis (coefficients -> physical) matrix.
    """
    m = x.shape[0]
    V = np.empty((m, n_modes), dtype=x.dtype)
    if n_modes >= 1:
        V[:, 0] = 1.0
    if n_modes >= 2:
        V[:, 1] = x
    for k in range(2, n_modes):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V


def _chebyshev_deriv_boundary_vectors(n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact analytic derivative of T_i at the two Chebyshev reference-domain
    endpoints: T_i'(-1) = (-1)^(i+1) * i^2, T_i'(1) = i^2 (with T_0' = 0).
    Used for the flux (C1) boundary evaluation.
    """
    i = np.arange(n_modes, dtype=np.float32)
    right = i ** 2
    left = ((-1.0) ** (i + 1.0)) * (i ** 2)
    return left, right


def _chebyshev_coefficient_derivative_matrix(n_modes: int, dtype=np.float64) -> np.ndarray:
    """
    Exact linear map ``D`` (``n_modes x n_modes``) taking the Chebyshev
    coefficients of ``u(xi)`` (reference domain ``xi`` in ``[-1, 1]``) to
    the Chebyshev coefficients of ``du/dxi`` — i.e. this differentiates
    *in coefficient space*, not by finite-differencing physical-space
    samples. Closed form (see e.g. Boyd, "Chebyshev and Fourier Spectral
    Methods", or Trefethen's derivation of the Chebyshev differentiation
    recurrence):

        a_k^{(1)} = (2 / c_k) * sum_{j > k, (j - k) odd} j * a_j,
        c_0 = 2, c_k = 1 for k >= 1.

    i.e. ``D[k, j] = 2*j/c_k`` when ``j > k`` and ``(j - k)`` is odd, else 0
    (upper-triangular, strictly above the diagonal, alternating-stride
    nonzero pattern).

    Verified against ``numpy.polynomial.chebyshev.chebder`` for every
    basis vector ``T_0..T_{n_modes-1}`` before use here (exact match to
    float64 precision) — in particular ``D @ e_1 == e_0`` (d/dx[x] = 1)
    and ``D @ e_2 == 4*e_1`` (d/dx[2x^2-1] = 4x).

    (Pure NumPy — this is host-side setup math with no gradient ever
    flowing through it, computed once per model construction regardless
    of which backend trains the model, so it doesn't need to be built in
    any deep-learning framework's tensor type at all.)
    """
    D = np.zeros((n_modes, n_modes), dtype=dtype)
    for k in range(n_modes):
        c_k = 2.0 if k == 0 else 1.0
        for j in range(k + 1, n_modes):
            if (j - k) % 2 == 1:
                D[k, j] = 2.0 * j / c_k
    return D


def build_separable_linear_operator(
    n_elements: int,
    dims: int,
    n_modes: int,
    dim_coeffs: List[Dict[int, float]],
    element_widths: Optional[Any] = None,
) -> np.ndarray:
    """
    Build the ``(n_elements, dims, n_modes, n_modes)`` linear-operator
    tensor consumed by ``USENOCPModule`` (its ``element_operators``
    argument), for any PDE whose linear part is *separable* — a sum of
    terms, each a pure derivative (or the identity) acting along exactly
    one spatial axis:

        L[u] = sum_d  coeff_{d,0} * u  +  coeff_{d,1} * du/dx_d
                    +  coeff_{d,2} * d^2u/dx_d^2  + ...

    Parameters
    ----------
    n_elements, dims, n_modes : as in ``USENOCPModule``.
    dim_coeffs : length-``dims`` list of ``{derivative_order: coefficient}``
        dicts, one per spatial axis. E.g. for 1D heat ``u_t = alpha * u_xx``
        (only the spatial operator goes here — the time derivative is
        handled outside this spatial operator), pass ``dim_coeffs=[{2: alpha}]``.
        This helper applies spatial derivatives only; it does not construct
        a time derivative. For 2D
        anisotropic diffusion plus a reaction term ``c*u + a*u_xx + b*u_yy``,
        pass ``dim_coeffs=[{0: c, 2: a}, {2: b}]``.
    element_widths : optional ``(n_elements,)`` array of physical element
        widths. These widths are used for the ``2.0 / width`` chain-rule
        scaling in both spatial operators and boundary fluxes. If omitted
        here, this helper uses unit widths; the trainer builder instead
        derives widths from the configured spatial domain. What matters for
        correctness is only that this scaling and ``USENOCPModule``'s
        boundary-flux scaling agree — which they do here by construction,
        since both use the identical ``2.0 / width`` factor — not the
        absolute physical meaning of "width" in isolation.

    IMPORTANT — separability limit on zeroth-order (reaction) terms:
    ``USENOCPModule.compute_residuals_and_boundaries`` builds
    ``sum_{d_op} (identity on every axis except d_op, L applied along
    d_op)`` — i.e. it sums one term *per axis*, each acting along a single
    designated axis and passing every other axis through untouched. A
    zeroth-order (reaction, ``c*u``) term has no derivative to "belong" to
    any one axis; if you put a nonzero order-0 entry in *every* axis's
    dict to represent one reaction term, it gets added once per axis and
    the reaction contribution is silently multiplied by ``dims``. The fix
    used throughout this codebase (see ``build_spectral_element_trainer``
    callers) is to fold any zeroth-order term into exactly ONE axis's
    dict — conventionally axis 0 — and leave order 0 out of every other
    axis's dict entirely. This function does not (and cannot, in general)
    detect or correct a caller who violates this convention: it faithfully
    builds whatever per-axis operator ``dim_coeffs`` describes, so getting
    the reaction term counted exactly once is the caller's responsibility.

    Higher-order derivative operators are built by repeated matrix
    multiplication of the coefficient-space derivative matrix (see
    ``_chebyshev_coefficient_derivative_matrix``); at truncated mode count
    ``n_modes``, this is nilpotent (``D**n_modes == 0`` exactly, since each
    application strictly increases minimum coefficient index), which is
    the expected/correct spectral-truncation behavior, not a bug.
    """
    if len(dim_coeffs) != dims:
        raise ValueError(
            f"build_separable_linear_operator: dim_coeffs must have exactly "
            f"`dims`={dims} entries (one per spatial axis), got {len(dim_coeffs)}."
        )

    if element_widths is None:
        widths64 = np.ones(n_elements, dtype=np.float64)
    else:
        widths64 = _to_numpy_host(element_widths).astype(np.float64)
        if widths64.shape[0] != n_elements:
            raise ValueError(
                f"build_separable_linear_operator: element_widths has "
                f"{widths64.shape[0]} entries, expected n_elements={n_elements}."
            )

    max_order = 0
    for coeffs in dim_coeffs:
        if coeffs:
            max_order = max(max_order, max(coeffs.keys()))

    D_ref = _chebyshev_coefficient_derivative_matrix(n_modes)  # reference-domain, xi in [-1, 1]
    identity = np.eye(n_modes, dtype=np.float64)

    # Reference-domain derivative operators D^0 .. D^{max_order}, each an
    # exact matrix power of the single first-derivative matrix.
    D_powers = [identity]
    current = identity
    for _ in range(max_order):
        current = current @ D_ref
        D_powers.append(current)

    operators = np.zeros((n_elements, dims, n_modes, n_modes), dtype=np.float64)
    for e in range(n_elements):
        width = widths64[e]
        for d in range(dims):
            op = np.zeros((n_modes, n_modes), dtype=np.float64)
            for order, coeff in dim_coeffs[d].items():
                if order < 0 or not float(order).is_integer():
                    raise ValueError(
                        f"build_separable_linear_operator: derivative order must be "
                        f"a non-negative integer, got {order} on axis {d}."
                    )
                # Chain rule for the affine reference-to-physical map:
                # d/dx_phys = (2/width) * d/dxi_ref, applied `order` times.
                phys_scale = (2.0 / width) ** order
                op = op + coeff * phys_scale * D_powers[order]
            operators[e, d] = op

    return operators.astype(np.float32)


# ---------------------------------------------------------------------------
# USENO CP-factored spectral-element neural operator
# ---------------------------------------------------------------------------

class USENOCPModule:
    """
    Unified Spectral Element Neural Operator mapped into CP Rank Space.
    Scales strictly linearly with dimensions up to 10D and beyond.

    Every per-dimension spectral series is represented in a *Chebyshev*
    polynomial basis (matching the analytic boundary evaluation vectors
    below), so any physical-space round-trip performed on those
    coefficients — e.g. for pseudospectral nonlinear-term evaluation —
    must use a Chebyshev synthesis/analysis pair, not a Fourier one.

    Backend dispatch
    ----------------
    Calling ``USENOCPModule(...)`` returns an instance of
    ``_USENOCPModuleTorch`` (which also subclasses ``torch.nn.Module``,
    so ``.parameters()``/``.named_buffers()``/device movement all keep
    working exactly as before) when ``backend.name == "torch"``, or
    ``_USENOCPModuleGeneric`` (a plain object — trainable state lives
    entirely in ``self.network``, a backend-native model from
    ``backend.build_mlp``) for every other backend. This is done via
    ``__new__`` rather than a fixed base class because Python decides a
    class's bases at class-*definition* time, before we know which
    backend a given call wants — see the two concrete subclasses right
    below this class. Both subclasses inherit every method here
    (``isinstance(model, USENOCPModule)`` always holds), so this is the
    class to type-hint against; only reach for the concrete subclasses
    if you specifically need to check for a PyTorch ``nn.Module`` instance.

    Prediction
    ----------
    ``model_fn`` accepts one row per query in ``[spatial coordinates..., t]``
    order and returns a scalar field value per row. Evaluation uses the
    configured spatial bounds and the model's element widths; the builder
    attaches those bounds when constructing a trainer. A standalone model
    must have ``spatial_bounds`` set before calling ``model_fn``.

    Parameters
    ----------
    n_elements, dims, n_modes, rank : as before.
    element_operators : (Elements, Dims, N_modes, N_modes) or
        (Elements, Dims, N_modes) linear spectral operators.
    element_widths : optional (Elements,) array of physical element widths
        along dimension 0, which is the axis used for interface stitching.
        The reference derivative at ``xi = +/-1`` is scaled by ``2 / width``
        to convert it to a physical derivative. The standalone module
        defaults to unit widths; ``build_spectral_element_trainer`` derives
        widths from the configured first spatial bound when omitted.
    """

    def __new__(cls, *args, backend: Optional[AbstractBackend] = None, **kwargs):
        if cls is not USENOCPModule:
            # A concrete subclass was instantiated directly — respect it.
            return super().__new__(cls)
        b = backend or get_backend("torch")
        if b.name == "torch":
            if not _TORCH_AVAILABLE:
                raise ImportError(
                    "USENOCPModule was requested with the torch backend, "
                    "but PyTorch is not installed in this environment."
                )
            return super().__new__(_USENOCPModuleTorch)
        return super().__new__(_USENOCPModuleGeneric)

    def __init__(
        self,
        n_elements: int,
        dims: int,
        n_modes: int,
        rank: int,
        element_operators,
        element_widths: Optional[Any] = None,
        backend: Optional[AbstractBackend] = None,
    ):
        # Run torch.nn.Module's own __init__ first (sets up the internal
        # _parameters/_buffers/_modules dicts register_buffer needs)
        # when this instance is the torch subclass. Done explicitly
        # rather than via super().__init__() chaining, since the two
        # concrete subclasses have different MROs and this needs to be
        # unconditionally correct for both.
        if _TORCH_AVAILABLE and isinstance(self, _nn.Module):
            _nn.Module.__init__(self)

        self.backend = backend or get_backend("torch")
        b = self.backend
        self.n_elements = n_elements
        self.dims = dims
        self.n_modes = n_modes
        self.rank = rank
        has_register_buffer = hasattr(self, "register_buffer")

        # Structural shape: (Elements, Dims, N_modes, N_modes)
        def store_tensor(name: str, value) -> None:
            # All setup constants below (linear operators, boundary
            # vectors, Chebyshev synthesis/analysis matrices) are built
            # as plain NumPy by the module-level helpers — see their
            # docstrings — so the only conversion needed here is NumPy
            # -> the active backend's own tensor type. _to_numpy_host
            # also transparently accepts a caller-supplied torch/jax/tf/
            # paddle tensor for element_operators/element_widths (kept
            # for backward compatibility with existing callers).
            tensor = b.tensor(_to_numpy_host(value))
            if has_register_buffer:
                self.register_buffer(name, tensor)
            else:
                setattr(self, name, tensor)

        store_tensor("linear_operators", element_operators)

        # Chebyshev boundary reconstruction vectors (value / C0).
        # At x=-1, T_i(-1) = (-1)^i. At x=1, T_i(1) = 1.
        left_eval = [(-1.0) ** i for i in range(n_modes)]
        right_eval = [1.0] * n_modes
        store_tensor("left_boundary_vector", left_eval)
        store_tensor("right_boundary_vector", right_eval)

        # Chebyshev boundary derivative vectors (flux / C1) — exact
        # analytic derivatives of T_i at the reference-domain endpoints.
        left_deriv, right_deriv = _chebyshev_deriv_boundary_vectors(n_modes)
        store_tensor("left_boundary_deriv_vector", left_deriv)
        store_tensor("right_boundary_deriv_vector", right_deriv)

        if element_widths is None:
            element_widths = np.ones(n_elements, dtype=np.float32)
        store_tensor("element_widths", element_widths)

        # Chebyshev analysis/synthesis matrices for the pseudospectral
        # nonlinear-term product, replacing the previous (incorrect)
        # Fourier FFT round-trip. Sized with a 2x oversampling pad, same
        # dealiasing margin the old FFT implementation used. Pure NumPy
        # (host-side setup math, no gradients) — including the
        # pseudo-inverse, via np.linalg.pinv rather than torch.linalg.pinv.
        pad_size = n_modes * 2
        nodes = _chebyshev_gauss_lobatto_nodes(pad_size)
        synth = _chebyshev_vandermonde(nodes, n_modes).astype(np.float32)   # (pad_size, n_modes)
        analyze = np.linalg.pinv(synth).astype(np.float32)                  # (n_modes, pad_size)
        store_tensor("cheb_synth", synth)
        store_tensor("cheb_analyze", analyze)

        # Deep Coefficient Factor Generation Backbone Network
        self.network = b.build_mlp(
            (1, 512, 1024, 1024, self.n_elements * self.dims * self.rank * self.n_modes),
            activation="tanh",
        )
        self._jax_params = None
        if b.name == "jax" and hasattr(self.network, "init"):
            import jax
            dummy = b.zeros((1, 1))
            self._jax_params = self.network.init(
                getattr(self.network, "init_key", jax.random.PRNGKey(0)), dummy
            )
            if hasattr(self.network, "bound_params"):
                self.network.bound_params = self._jax_params

    def parameters(self, recurse: bool = True):
        if self.backend.name == "torch":
            return super().parameters(recurse=recurse)
        return self.backend.parameters(self.network)

    def named_parameters(self, *args, **kwargs):
        if self.backend.name == "torch":
            return super().named_parameters(*args, **kwargs)
        return {f"param_{i}": p for i, p in enumerate(self.parameters())}

    def init_jax_params(self, dummy_input):
        if self._jax_params is None:
            self._jax_params = getattr(self.network, "params", None)
        return self._jax_params

    def set_jax_params(self, params):
        self._jax_params = params
        if hasattr(self.network, "bound_params"):
            self.network.bound_params = params

    def get_cp_factors(self, t: BackendTensor) -> List[List[BackendTensor]]:
        """
        Decodes network out-stream into structure-safe 1D Factor groups.
        Structure: output[element_idx][dimension_idx] -> Tensor(Batch, Rank, N_modes)
        """
        batch_size = t.shape[0]
        b = self.backend
        if b.name == "jax":
            params = self._jax_params if self._jax_params is not None else getattr(self.network, "params", None)
            raw_output = self.network.apply(params, t)
        elif b.name == "tensorflow":
            raw_output = self.network(t, training=False)
        else:
            raw_output = self.network(t)

        # Reshape to element tensor format
        reshaped = b.reshape(raw_output, (batch_size, self.n_elements, self.dims, self.rank, self.n_modes))

        elements_factors = []
        for e in range(self.n_elements):
            dim_list = []
            for d in range(self.dims):
                dim_list.append(reshaped[:, e, d, :, :])
            elements_factors.append(dim_list)

        return elements_factors

    def pseudospectral_1d_product(
        self,
        factor_1: BackendTensor,
        factor_2: BackendTensor,
        nonlinear_fn: Optional[Callable[[BackendTensor, BackendTensor], BackendTensor]] = None,
    ) -> BackendTensor:
        """
        Performs 1D pseudospectral nonlinear-term multiplication in the
        Chebyshev basis:

            Chebyshev coefficients
                -> synthesis (Vandermonde eval at Chebyshev-Gauss-Lobatto
                   nodes, ``self.cheb_synth``)
                -> physical-space multiply
                -> analysis (least-squares Chebyshev fit,
                   ``self.cheb_analyze``, the pseudo-inverse of synthesis)
                -> Chebyshev coefficients

        This intentionally does NOT use an FFT — an FFT-based
        analysis/synthesis pair is only valid for a Fourier/trigonometric
        basis, and these factors are Chebyshev coefficients (see the
        module docstring for why mixing the two is a genuine correctness
        bug, not a style choice).

        ``nonlinear_fn``, if given, is applied to the two physical-space
        factors (``nonlinear_fn(spatial_1, spatial_2) -> spatial_result``)
        in place of the default plain product ``spatial_1 * spatial_2``.
        This is how callers plug in the actual nonlinear term of their PDE
        (e.g. Burgers' u*u, a cubic NLS nonlinearity, etc.) instead of
        always getting a generic self-product no matter what physics was
        requested.

        Shapes: (Batch, Rank, N_modes) or (Batch, N_modes).
        """
        n_modes = factor_1.shape[-1]
        synth = self.cheb_synth
        analyze = self.cheb_analyze

        # coefficients (..., n_modes) @ synth.T (n_modes, pad_size) -> (..., pad_size)
        spatial_1 = self.backend.matmul(factor_1, synth.T)
        spatial_2 = self.backend.matmul(factor_2, synth.T)

        if nonlinear_fn is not None:
            product_spatial = nonlinear_fn(spatial_1, spatial_2)
        else:
            product_spatial = spatial_1 * spatial_2

        # (..., pad_size) @ analyze.T (pad_size, n_modes) -> (..., n_modes)
        spectral_product = self.backend.matmul(product_spatial, analyze.T)
        return spectral_product[..., :n_modes]

    def evaluate(self, t, spatial_points, spatial_bounds):
        """Evaluate the CP-factorized field at paired time/space points.

        ``spatial_points`` has shape ``(batch, dims)`` and ``t`` has shape
        ``(batch, 1)``.  Elements partition the first spatial axis; their
        widths are physical lengths and must span ``spatial_bounds[0]``.
        The returned tensor has shape ``(batch, 1)``.
        """
        b = self.backend
        if len(spatial_bounds) != self.dims:
            raise ValueError(
                f"spatial_bounds must contain {self.dims} axes, got {len(spatial_bounds)}."
            )
        if spatial_points.ndim != 2 or spatial_points.shape[1] != self.dims:
            raise ValueError(
                f"spatial_points must have shape (batch, {self.dims}), "
                f"got {tuple(spatial_points.shape)}."
            )
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim != 2 or t.shape[1] != 1 or t.shape[0] != spatial_points.shape[0]:
            raise ValueError("t must have shape (batch, 1) matching spatial_points.")

        factors = self.get_cp_factors(t)
        points_np = b.to_numpy(spatial_points)
        widths = b.to_numpy(self.element_widths).reshape(-1)
        x_lower, x_upper = map(float, spatial_bounds[0])
        if any(float(hi) <= float(lo) for lo, hi in spatial_bounds):
            raise ValueError("Each spatial bound must have positive width.")
        if widths.shape != (self.n_elements,) or np.any(widths <= 0):
            raise ValueError("element_widths must contain one positive width per element.")
        tol = 1e-7 * max(1.0, abs(x_lower), abs(x_upper))
        if np.any(points_np[:, 0] < x_lower - tol) or np.any(points_np[:, 0] > x_upper + tol):
            raise ValueError("spatial_points lie outside the first spatial bound.")
        if not np.isclose(widths.sum(), x_upper - x_lower, rtol=1e-5, atol=tol):
            raise ValueError(
                "element_widths must sum to the width of spatial_bounds[0] "
                "to evaluate the spectral elements."
            )

        result = b.zeros((spatial_points.shape[0], 1), dtype=spatial_points.dtype)
        left = x_lower
        for e, width in enumerate(widths):
            right = left + float(width)
            is_last = e == self.n_elements - 1
            mask_np = (points_np[:, 0] >= left - tol) & (
                (points_np[:, 0] <= right + tol) if is_last else (points_np[:, 0] < right)
            )
            mask = b.tensor(mask_np[:, None], device=b.default_device())
            mask = b.cast(mask, result.dtype)
            xi = (2.0 * (spatial_points[:, 0] - left) / float(width) - 1.0)[:, None]

            value = None
            for d in range(self.dims):
                if d == 0:
                    coordinate = xi[:, 0]
                else:
                    lo, hi = map(float, spatial_bounds[d])
                    coordinate = (2.0 * (spatial_points[:, d] - lo) / (hi - lo) - 1.0)
                # Evaluate T_0 ... T_(n_modes-1) without leaving the active
                # backend, preserving compatibility with autograd tensors.
                basis = [b.ones((coordinate.shape[0],), dtype=coordinate.dtype)]
                if self.n_modes > 1:
                    basis.append(coordinate)
                for _ in range(2, self.n_modes):
                    basis.append(2.0 * coordinate * basis[-1] - basis[-2])
                vandermonde = b.stack(basis, axis=1)
                dimension_value = b.sum(
                    factors[e][d] * vandermonde[:, None, :], axis=-1,
                )
                value = dimension_value if value is None else value * dimension_value
            element_value = b.sum(value, axis=1, )[:, None]
            result = result + mask * element_value
            left = right
        return result

    def model_fn(self, points):
        """Evaluate points in the package-wide ``[space..., time]`` layout."""
        bounds = getattr(self, "spatial_bounds", None)
        if bounds is None:
            raise RuntimeError(
                "Spatial bounds are required for USENO prediction; construct "
                "the model through Trainer.for_spectral_element()."
            )
        if points.ndim != 2 or points.shape[1] != self.dims + 1:
            raise ValueError(
                f"points must have shape (batch, {self.dims + 1}) "
                "with spatial coordinates followed by time."
            )
        return self.evaluate(points[:, -1:], points[:, :self.dims], bounds)

    def compute_residuals_and_boundaries(
        self, t: BackendTensor, nonlinear_fn: Optional[Callable],
    ) -> Tuple[List[List[BackendTensor]], List[Dict[str, List[BackendTensor]]]]:
        """
        Constructs the structural components of the PDE terms using CP representations.

        ``nonlinear_fn(spatial_1, spatial_2) -> spatial_result`` defines the
        pseudospectral nonlinear interaction used for every rank-pair
        product; pass ``None`` to fall back to the plain product
        ``spatial_1 * spatial_2``.

        Returns:
            linear_residual_cp: List of CP factor lists per element
            boundary_data: per element, a dict with:
                "left" / "right"             — value (C0) boundary factors
                "left_deriv" / "right_deriv" — flux (C1) boundary factors
        """
        elements_factors = self.get_cp_factors(t)

        all_elements_residuals = []
        boundary_evals = []

        for e in range(self.n_elements):
            current_factors = elements_factors[e]
            # Storage structures representing the compiled residual CP components
            # For linear terms: D_d applied to d-th factor, others untouched. Total elements = dims * rank
            residual_factors = [[] for _ in range(self.dims)]

            # 1. Compute Linear Operators Over Dimensions
            for d_op in range(self.dims):
                op_matrix = self.linear_operators[e, d_op]  # Size (N_modes, N_modes) or (N_modes,)

                for r in range(self.rank):
                    for d_field in range(self.dims):
                        f = current_factors[d_field][:, r, :]  # (Batch, N_modes)
                        if d_field == d_op:
                            if op_matrix.ndim == 1:
                                # Diagonal spectral operator: use element-wise multiplication
                                f_transformed = f * op_matrix
                            else:
                                # Full transformation matrix: use batched matrix contraction
                                f_transformed = self.backend.matmul(f, op_matrix.T)

                            residual_factors[d_field].append(f_transformed)
                        else:
                            residual_factors[d_field].append(f)

            # 2. Compute Non-linear Term Expansion (e.g., u * u via pseudospectral 1D interactions)
            # Produces a rank expansion to rank^2 terms
            for r1 in range(self.rank):
                for r2 in range(self.rank):
                    for d_field in range(self.dims):
                        f1 = current_factors[d_field][:, r1, :]
                        f2 = current_factors[d_field][:, r2, :]
                        nl_f = self.pseudospectral_1d_product(f1, f2, nonlinear_fn=nonlinear_fn)
                        residual_factors[d_field].append(nl_f)

            # Stack compiled factor lists to form verified CP matrices: (Batch, Total_Rank, N_modes)
            compiled_residual = [self.backend.stack(residual_factors[d], axis=1) for d in range(self.dims)]
            all_elements_residuals.append(compiled_residual)

            # 3. Extract Global Boundary Intersections for Continuity Stitching
            # Compute value (C0) AND flux (C1) evaluation maps across
            # primary boundaries along dimension axis 0.
            width_e = self.element_widths[e]
            deriv_scale = 2.0 / width_e  # chain rule: d/dx_phys = (2/width) * d/dxi_ref

            left_boundary_factors = []
            right_boundary_factors = []
            left_deriv_factors = []
            right_deriv_factors = []
            for d_field in range(self.dims):
                if d_field == 0:
                    # Contract dimension axis 0 against boundary evaluation nodes
                    l_val = self.backend.matmul(current_factors[d_field], self.left_boundary_vector)[..., None]
                    r_val = self.backend.matmul(current_factors[d_field], self.right_boundary_vector)[..., None]
                    left_boundary_factors.append(l_val)
                    right_boundary_factors.append(r_val)

                    l_dv = deriv_scale * self.backend.matmul(
                        current_factors[d_field], self.left_boundary_deriv_vector
                    )[..., None]
                    r_dv = deriv_scale * self.backend.matmul(
                        current_factors[d_field], self.right_boundary_deriv_vector
                    )[..., None]
                    left_deriv_factors.append(l_dv)
                    right_deriv_factors.append(r_dv)
                else:
                    # Remaining dimensions cross untouched to form the (D-1) dimensional manifold
                    left_boundary_factors.append(current_factors[d_field])
                    right_boundary_factors.append(current_factors[d_field])
                    left_deriv_factors.append(current_factors[d_field])
                    right_deriv_factors.append(current_factors[d_field])

            boundary_evals.append({
                "left": left_boundary_factors,
                "right": right_boundary_factors,
                "left_deriv": left_deriv_factors,
                "right_deriv": right_deriv_factors,
            })

        return all_elements_residuals, boundary_evals


class _USENOCPModuleGeneric(USENOCPModule):
    """Concrete USENOCPModule for every non-torch backend (jax, tensorflow,
    paddle): a plain object, since trainable state lives entirely in
    ``self.network`` and none of this needs torch's parameter/buffer
    bookkeeping. See USENOCPModule's docstring for the dispatch rationale."""
    pass


if _TORCH_AVAILABLE:
    class _USENOCPModuleTorch(USENOCPModule, _nn.Module):
        """Concrete USENOCPModule for the torch backend: also a real
        torch.nn.Module, so .parameters()/.named_buffers()/device movement
        all work exactly as they did before this class supported other
        backends. See USENOCPModule's docstring for the dispatch rationale.

        NOTE ON BASE ORDER: USENOCPModule must come FIRST here. Both
        USENOCPModule and nn.Module define __init__; Python resolves
        `_USENOCPModuleTorch.__init__` to whichever appears first in the
        MRO, so (nn.Module, USENOCPModule) would silently call
        nn.Module.__init__(self) instead of our real constructor. With
        USENOCPModule first, USENOCPModule.__init__ runs (and explicitly
        calls _nn.Module.__init__(self) itself as its first line — see
        that method), while isinstance(model, nn.Module) and method
        lookups for register_buffer/parameters/etc. are unaffected by
        base order and still resolve onto nn.Module as expected."""
        pass
else:
    # Referenced (but never actually reachable) from USENOCPModule.__new__
    # when backend.name == "torch" — that branch raises ImportError before
    # this name would ever need to resolve to a real class.
    _USENOCPModuleTorch = None


__all__ = [
    "TensorCPMath", "DynamicLossBalancer", "USENOCPModule",
    "build_separable_linear_operator",
]
