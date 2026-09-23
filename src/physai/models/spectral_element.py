"""
physai/models/spectral_element.py

Unified Spectral Element Neural Operator (USENO), mapped into CP
(Canonical Polyadic) rank space, following the reference architecture in
Feugmo & Pankaczy, "Neural Spectral Element Methods for stiff multiphysics
PDEs" (arXiv:2606.02335).

This module used to live inside ``physai.solvers.dedalus`` alongside the
*real* Dedalus ``Solver`` class, which was a naming/scoping bug in its own
right: none of ``TensorCPMath`` / ``DynamicLossBalancer`` / ``USENOCPModule``
touch the ``dedalus`` package at all — they are pure PyTorch. That file has
been split so that ``physai.solvers.dedalus`` is Dedalus-only, and this
module (imported by ``physai.models.spectralpinn``) is the pure-PyTorch
neural-operator side.

Two correctness bugs, found by checking the implementation against the
reference paper, were also fixed as part of the split:

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

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable, Optional


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
    def compute_cp_inner_product(factors_A: List[torch.Tensor], factors_B: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the inner product between two CP tensors A and B across batches.
        Formula: <A, B> = Sum_{r1, r2} Prod_d <factors_A[d][:, r1], factors_B[d][:, r2]>
        Shapes: factors_A[d] -> (Batch, Rank_A, N_modes)
                factors_B[d] -> (Batch, Rank_B, N_modes)
        Returns: (Batch,) tensor of inner products.
        """
        batch_size = factors_A[0].shape[0]
        rank_A = factors_A[0].shape[1]
        rank_B = factors_B[0].shape[1]
        dims = len(factors_A)

        # Initialize the interaction matrix with ones: shape (Batch, Rank_A, Rank_B)
        interaction = torch.ones((batch_size, rank_A, rank_B), device=factors_A[0].device, dtype=factors_A[0].dtype)

        for d in range(dims):
            # Handle complex inner products via conjugation.
            if factors_B[d].is_complex():
                b_matrix = factors_B[d].conj().transpose(1, 2)
            else:
                b_matrix = factors_B[d].transpose(1, 2)

            # (Batch, Rank_A, N_modes) x (Batch, N_modes, Rank_B) -> (Batch, Rank_A, Rank_B)
            dim_inner = torch.bmm(factors_A[d], b_matrix)
            interaction = interaction * dim_inner

        # Sum over all rank combinations to get the final scalar inner product per batch item
        result = torch.sum(interaction, dim=(1, 2))

        # If the result is still carrying a 0j complex tail due to precision, cast to real
        if result.is_complex():
            return result.real
        return result

    @staticmethod
    def compute_cp_norm_squared(factors: List[torch.Tensor]) -> torch.Tensor:
        """Computes the squared L2 norm of a CP tensor: <factors, factors>"""
        return TensorCPMath.compute_cp_inner_product(factors, factors)


# ---------------------------------------------------------------------------
# Dynamic multi-objective loss balancing
# ---------------------------------------------------------------------------

class DynamicLossBalancer:
    """
    Industrial gradient-matching system (Modified Neural Tangent Kernel alignment).
    Dynamically balances weights of multi-objective losses to prevent gradient pathologies.
    """
    def __init__(self, num_losses: int, alpha: float = 0.9):
        self.num_losses = num_losses
        self.alpha = alpha
        self.weights = torch.ones(num_losses, dtype=torch.float32)

    def update_weights(self, losses: List[torch.Tensor], shared_parameters: List[torch.nn.Parameter]):
        """Adjusts objective loss weights based on the standard deviation of their gradients."""
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

def _chebyshev_gauss_lobatto_nodes(m: int) -> torch.Tensor:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1], x_j = cos(pi * j / (m - 1))."""
    j = torch.arange(m, dtype=torch.float64)
    return torch.cos(torch.pi * j / (m - 1))


def _chebyshev_vandermonde(x: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    Evaluate T_0..T_{n_modes-1} at points ``x`` via the standard three-term
    recurrence T_0=1, T_1=x, T_{k+1}=2x*T_k - T_{k-1}.

    Returns a (len(x), n_modes) matrix V with V[j, i] = T_i(x_j) — the exact
    Chebyshev synthesis (coefficients -> physical) matrix.
    """
    m = x.shape[0]
    V = torch.empty((m, n_modes), dtype=x.dtype)
    if n_modes >= 1:
        V[:, 0] = 1.0
    if n_modes >= 2:
        V[:, 1] = x
    for k in range(2, n_modes):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V


def _chebyshev_deriv_boundary_vectors(n_modes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact analytic derivative of T_i at the two Chebyshev reference-domain
    endpoints: T_i'(-1) = (-1)^(i+1) * i^2, T_i'(1) = i^2 (with T_0' = 0).
    Used for the flux (C1) boundary evaluation.
    """
    i = torch.arange(n_modes, dtype=torch.float32)
    right = i ** 2
    left = ((-1.0) ** (i + 1.0)) * (i ** 2)
    return left, right


def _chebyshev_coefficient_derivative_matrix(n_modes: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
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
    """
    D = torch.zeros((n_modes, n_modes), dtype=dtype)
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
    element_widths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
        handled by ``USENOPINNTrainer``'s own time marching, not by this
        spatial operator), pass ``dim_coeffs=[{2: alpha}]``. For 2D
        anisotropic diffusion plus a reaction term ``c*u + a*u_xx + b*u_yy``,
        pass ``dim_coeffs=[{0: c, 2: a}, {2: b}]``.
    element_widths : optional ``(n_elements,)`` tensor forwarded to the
        same ``2.0 / width`` chain-rule scaling ``USENOCPModule`` applies
        to its own boundary flux vectors (see that class's docstring for
        the exact convention it uses for "width"). Defaults to all-ones,
        matching ``USENOCPModule``'s own default. What matters for
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
        widths64 = torch.ones(n_elements, dtype=torch.float64)
    else:
        if element_widths.shape[0] != n_elements:
            raise ValueError(
                f"build_separable_linear_operator: element_widths has "
                f"{element_widths.shape[0]} entries, expected n_elements={n_elements}."
            )
        widths64 = element_widths.to(torch.float64)

    max_order = 0
    for coeffs in dim_coeffs:
        if coeffs:
            max_order = max(max_order, max(coeffs.keys()))

    D_ref = _chebyshev_coefficient_derivative_matrix(n_modes)  # reference-domain, xi in [-1, 1]
    identity = torch.eye(n_modes, dtype=torch.float64)

    # Reference-domain derivative operators D^0 .. D^{max_order}, each an
    # exact matrix power of the single first-derivative matrix.
    D_powers = [identity]
    current = identity
    for _ in range(max_order):
        current = current @ D_ref
        D_powers.append(current)

    operators = torch.zeros((n_elements, dims, n_modes, n_modes), dtype=torch.float64)
    for e in range(n_elements):
        width = widths64[e]
        for d in range(dims):
            op = torch.zeros((n_modes, n_modes), dtype=torch.float64)
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

    return operators.to(torch.float32)


# ---------------------------------------------------------------------------
# USENO CP-factored spectral-element neural operator
# ---------------------------------------------------------------------------

class USENOCPModule(nn.Module):
    """
    Unified Spectral Element Neural Operator mapped into CP Rank Space.
    Scales strictly linearly with dimensions up to 10D and beyond.

    Every per-dimension spectral series is represented in a *Chebyshev*
    polynomial basis (matching the analytic boundary evaluation vectors
    below), so any physical-space round-trip performed on those
    coefficients — e.g. for pseudospectral nonlinear-term evaluation —
    must use a Chebyshev synthesis/analysis pair, not a Fourier one.

    Parameters
    ----------
    n_elements, dims, n_modes, rank : as before.
    element_operators : (Elements, Dims, N_modes, N_modes) or
        (Elements, Dims, N_modes) linear spectral operators.
    element_widths : optional (Elements,) tensor giving each element's
        physical half-width along dimension 0 (the axis boundary
        stitching acts on). The reference-domain derivative computed from
        ``T_i'(±1)`` must be scaled by ``2 / width`` (chain rule for the
        affine map from the physical element onto the [-1, 1] reference
        domain) to become a genuine physical flux before comparing across
        elements of different size. Defaults to all-ones (reference space
        == physical space), which reproduces the un-scaled behaviour for
        uniform-width elements.
    """
    def __init__(
        self,
        n_elements: int,
        dims: int,
        n_modes: int,
        rank: int,
        element_operators: torch.Tensor,
        element_widths: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.dims = dims
        self.n_modes = n_modes
        self.rank = rank

        # Structural shape: (Elements, Dims, N_modes, N_modes)
        self.register_buffer("linear_operators", element_operators)

        # Chebyshev boundary reconstruction vectors (value / C0).
        # At x=-1, T_i(-1) = (-1)^i. At x=1, T_i(1) = 1.
        left_eval = torch.tensor([(-1.0) ** i for i in range(n_modes)], dtype=torch.float32)
        right_eval = torch.ones(n_modes, dtype=torch.float32)
        self.register_buffer("left_boundary_vector", left_eval)
        self.register_buffer("right_boundary_vector", right_eval)

        # Chebyshev boundary derivative vectors (flux / C1) — exact
        # analytic derivatives of T_i at the reference-domain endpoints.
        left_deriv, right_deriv = _chebyshev_deriv_boundary_vectors(n_modes)
        self.register_buffer("left_boundary_deriv_vector", left_deriv)
        self.register_buffer("right_boundary_deriv_vector", right_deriv)

        if element_widths is None:
            element_widths = torch.ones(n_elements, dtype=torch.float32)
        self.register_buffer("element_widths", element_widths.to(torch.float32))

        # Chebyshev analysis/synthesis matrices for the pseudospectral
        # nonlinear-term product, replacing the previous (incorrect)
        # Fourier FFT round-trip. Sized with a 2x oversampling pad, same
        # dealiasing margin the old FFT implementation used.
        pad_size = n_modes * 2
        nodes = _chebyshev_gauss_lobatto_nodes(pad_size)
        synth = _chebyshev_vandermonde(nodes, n_modes).to(torch.float32)          # (pad_size, n_modes)
        analyze = torch.linalg.pinv(synth)                                        # (n_modes, pad_size)
        self.register_buffer("cheb_synth", synth)
        self.register_buffer("cheb_analyze", analyze)

        # Deep Coefficient Factor Generation Backbone Network
        self.network = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, self.n_elements * self.dims * self.rank * self.n_modes)
        )

    def get_cp_factors(self, t: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Decodes network out-stream into structure-safe 1D Factor groups.
        Structure: output[element_idx][dimension_idx] -> Tensor(Batch, Rank, N_modes)
        """
        batch_size = t.shape[0]
        raw_output = self.network(t)

        # Reshape to element tensor format
        reshaped = raw_output.view(batch_size, self.n_elements, self.dims, self.rank, self.n_modes)

        elements_factors = []
        for e in range(self.n_elements):
            dim_list = []
            for d in range(self.dims):
                dim_list.append(reshaped[:, e, d, :, :])
            elements_factors.append(dim_list)

        return elements_factors

    def pseudospectral_1d_product(
        self,
        factor_1: torch.Tensor,
        factor_2: torch.Tensor,
        nonlinear_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
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

        This intentionally does NOT use ``torch.fft`` — an FFT-based
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
        synth = self.cheb_synth.to(dtype=factor_1.dtype)      # (pad_size, n_modes)
        analyze = self.cheb_analyze.to(dtype=factor_1.dtype)  # (n_modes, pad_size)

        # coefficients (..., n_modes) @ synth.T (n_modes, pad_size) -> (..., pad_size)
        spatial_1 = factor_1 @ synth.T
        spatial_2 = factor_2 @ synth.T

        if nonlinear_fn is not None:
            product_spatial = nonlinear_fn(spatial_1, spatial_2)
        else:
            product_spatial = spatial_1 * spatial_2

        # (..., pad_size) @ analyze.T (pad_size, n_modes) -> (..., n_modes)
        spectral_product = product_spatial @ analyze.T
        return spectral_product[..., :n_modes]

    def compute_residuals_and_boundaries(
        self, t: torch.Tensor, nonlinear_fn: Callable,
    ) -> Tuple[List[List[torch.Tensor]], List[Dict[str, List[torch.Tensor]]]]:
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
                                f_transformed = torch.matmul(f, op_matrix.t())

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
            compiled_residual = [torch.stack(residual_factors[d], dim=1) for d in range(self.dims)]
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
                    l_val = torch.matmul(current_factors[d_field], self.left_boundary_vector).unsqueeze(-1)
                    r_val = torch.matmul(current_factors[d_field], self.right_boundary_vector).unsqueeze(-1)
                    left_boundary_factors.append(l_val)
                    right_boundary_factors.append(r_val)

                    l_dv = deriv_scale * torch.matmul(
                        current_factors[d_field], self.left_boundary_deriv_vector
                    ).unsqueeze(-1)
                    r_dv = deriv_scale * torch.matmul(
                        current_factors[d_field], self.right_boundary_deriv_vector
                    ).unsqueeze(-1)
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


__all__ = [
    "TensorCPMath", "DynamicLossBalancer", "USENOCPModule",
    "build_separable_linear_operator",
]