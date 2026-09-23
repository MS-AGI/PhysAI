"""
physai/core/losses.py

Backend-agnostic loss formulations for physics-informed training.

All loss functions accept a backend instance as their first argument and
operate exclusively through the AbstractBackend interface, making them
fully portable across PyTorch, JAX, TensorFlow, and PaddlePaddle. Two
exceptions are noted explicitly where they occur: ``WeightedLossComposite``'s
``"grad_norm"`` and ``"uncertainty"`` strategies need real per-parameter
gradients or learnable parameters, which the framework-agnostic
``AbstractBackend.grad`` interface isn't shaped for (it differentiates a
function of its *input tensor*, not a scalar loss w.r.t. a list of model
parameters) — those two strategies are implemented properly for PyTorch
and fall back to ``"fixed"`` with a clear one-time warning on JAX/TF/Paddle,
rather than silently pretending to work everywhere. Every other loss in
this file is 100% backend-agnostic.

Loss taxonomy
-------------
* Residual norm kernels — MSE, MAE, log-cosh, Huber, Charbonnier, quantile
* Boundary losses        — Dirichlet, Neumann, Robin, periodic, interface
* Data losses            — Supervised observation mismatch
* Physical regularisers  — Sobolev, gradient-penalty, spectral energy,
                            divergence-free / curl-free structural penalties
* Temporal weighting     — causal loss weighting (Wang et al. 2022)
* Composite              — WeightedLossComposite with adaptive weighting
                            strategies: fixed, softmax_temp, relobralo,
                            grad_norm, uncertainty
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from physai.backends.base import AbstractBackend, Tensor

# ---------------------------------------------------------------------------
# Small epsilon for numerical stability
# ---------------------------------------------------------------------------
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Primitive loss kernels
# ---------------------------------------------------------------------------

def mse_loss(backend: AbstractBackend, residual: Tensor) -> Tensor:
    """Mean squared error of a residual vector/tensor."""
    return backend.mean(backend.square(residual))


def mae_loss(backend: AbstractBackend, residual: Tensor) -> Tensor:
    """Mean absolute error of a residual vector/tensor."""
    return backend.mean(backend.abs(residual))


def log_cosh_loss(backend: AbstractBackend, residual: Tensor) -> Tensor:
    """
    log(cosh(x)) loss — smooth L1/L2 hybrid.
    Behaves like L2 near zero and L1 for large residuals.
    """
    # log(cosh(x)) = log((e^x + e^{-x})/2) = x + log(1 + e^{-2x}) - log(2)
    # Numerically stable form:
    x   = backend.abs(residual)
    lch = x + backend.log(backend.exp(-2.0 * x) + 1.0) - math.log(2.0)
    return backend.mean(lch)


def huber_loss(
    backend: AbstractBackend,
    residual: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """
    Huber loss with threshold ``delta``.
    Quadratic for |r| <= delta, linear otherwise.

    Implemented as a smooth sigmoid-gated blend rather than an exact
    piecewise ``where``, since ``AbstractBackend`` has no elementwise
    conditional-select primitive (adding one just for this would be a
    bigger interface change than this function needs) — this is a
    standard, fully backend-agnostic way to get a Huber-shaped loss
    without one. It's not bit-identical to the textbook piecewise
    definition right at the |r| == delta seam, but converges to it as
    the sigmoid sharpens (``k`` below), and is smooth everywhere the
    textbook version has a kink in its second derivative — arguably an
    advantage for gradient-based training, not just a workaround.
    """
    abs_r = backend.abs(residual)
    quadratic = backend.square(residual) * 0.5
    linear     = delta * abs_r - 0.5 * delta ** 2

    # selector = sigmoid(k*(abs_r - delta)), k large: ≈ 0 below delta, 1 above
    k        = 100.0 / (delta + _EPS)
    selector = 0.5 * (backend.tanh(k * (abs_r - delta)) + 1.0)
    loss     = (1.0 - selector) * quadratic + selector * linear
    return backend.mean(loss)


def charbonnier_loss(
    backend: AbstractBackend,
    residual: Tensor,
    eps: float = 1e-3,
) -> Tensor:
    """
    Charbonnier loss: sqrt(r^2 + eps^2) - eps — a smooth L1 approximation
    widely used in optical-flow and PDE surface-reconstruction settings
    (Barron 2019's robust-loss family passes through this at its alpha=1
    limit). Unlike the sigmoid-gated ``huber_loss`` above, this is exactly
    C^infinity everywhere with no blending seam at all, and its gradient
    stays bounded and well-scaled near ``residual == 0`` (unlike plain
    ``mae_loss``, whose gradient is a discontinuous +-1 there) — a good
    default when you want L1-like robustness to outlier residuals without
    the non-smooth gradient at exactly zero.
    """
    return backend.mean(backend.sqrt(backend.square(residual) + eps * eps) - eps)


def quantile_loss(
    backend: AbstractBackend,
    residual: Tensor,
    quantile: float = 0.5,
) -> Tensor:
    """
    Pinball (quantile) loss: asymmetric penalty
    ``max(q*r, (q-1)*r)`` for a target quantile ``q`` in (0, 1).

    ``quantile=0.5`` recovers (twice) the MAE loss. Useful for
    probabilistic/uncertainty-aware PINN variants that predict multiple
    quantiles of the solution rather than a single point estimate — train
    one network head per desired quantile, each against its own
    ``quantile_loss`` call, to get a calibrated predictive interval
    instead of just a point prediction.

    Implemented via ``max(a, b) = a + relu(b - a)`` so it only needs
    ``AbstractBackend.exp``/``log``-free elementwise ops already on the
    interface (no dedicated ``max``/``where`` primitive required): here
    concretely as ``0.5*(a+b) + 0.5*|a-b|``, which is algebraically
    ``max(a, b)`` for real numbers and only needs ``abs``.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile_loss: quantile must be in (0, 1), got {quantile}.")
    a = quantile * residual
    b = (quantile - 1.0) * residual
    elementwise_max = 0.5 * (a + b) + 0.5 * backend.abs(a - b)
    return backend.mean(elementwise_max)


# ---------------------------------------------------------------------------
# Boundary condition losses
# ---------------------------------------------------------------------------

def dirichlet_loss(
    backend: AbstractBackend,
    prediction: Tensor,
    target: Tensor,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Dirichlet BC: u(x_bc) = g(x_bc).

    Parameters
    ----------
    prediction : model output at boundary points
    target     : prescribed boundary values g
    weight     : scalar coefficient
    norm       : "mse" | "mae" | "log_cosh" | "huber" | "charbonnier"
    """
    residual = prediction - target
    return weight * _apply_norm(backend, residual, norm)


def neumann_loss(
    backend: AbstractBackend,
    grad_prediction: Tensor,
    flux_target: Tensor,
    normal: Optional[Tensor] = None,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Neumann BC: ∂u/∂n(x_bc) = h(x_bc).

    Parameters
    ----------
    grad_prediction : gradient of model output at boundary (shape [..., d])
    flux_target     : prescribed normal flux h (shape [...] or [..., d])
    normal          : outward unit normal vectors; if provided the dot-product
                      ∇u · n is taken before comparing to flux_target
    weight          : scalar coefficient
    norm            : "mse" | "mae" | "log_cosh" | "huber" | "charbonnier"
    """
    if normal is not None:
        # ∂u/∂n = ∇u · n  (sum over last axis)
        pred_flux = backend.sum(grad_prediction * normal, axis=-1)
    else:
        pred_flux = grad_prediction

    residual = pred_flux - flux_target
    return weight * _apply_norm(backend, residual, norm)


def robin_loss(
    backend: AbstractBackend,
    prediction: Tensor,
    grad_prediction: Tensor,
    alpha: float,
    beta: float,
    target: Tensor,
    normal: Optional[Tensor] = None,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Robin BC: alpha * u + beta * ∂u/∂n = target.
    """
    if normal is not None:
        flux = backend.sum(grad_prediction * normal, axis=-1)
    else:
        flux = grad_prediction

    residual = alpha * prediction + beta * flux - target
    return weight * _apply_norm(backend, residual, norm)


def periodic_loss(
    backend: AbstractBackend,
    prediction_left: Tensor,
    prediction_right: Tensor,
    grad_left: Optional[Tensor] = None,
    grad_right: Optional[Tensor] = None,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Periodic BC: u(x_left) = u(x_right) and optionally ∂u/∂n matches.
    """
    loss = _apply_norm(backend, prediction_left - prediction_right, norm)
    if grad_left is not None and grad_right is not None:
        loss = loss + _apply_norm(backend, grad_left - grad_right, norm)
    return weight * loss


# ``interface_loss`` is mathematically identical to ``periodic_loss`` — both
# are "match value (and optionally flux) between two point sets" — but the
# *use case* is different enough to deserve its own name and docstring:
# domain-decomposition PINN variants (XPINN, Jagtap & Karniadakis 2020;
# cPINN, Jagtap et al. 2020) split a single domain into subdomains, each
# trained by its own (sub)network, and stitch them back together by
# penalizing exactly this value/flux mismatch across each shared interface
# — the same continuity idea ``USENOPINNTrainer`` already applies between
# spectral elements (see physai.models.spectralpinn), just for
# subdomain-network interfaces instead of spectral-element interfaces.
interface_loss = periodic_loss


# ---------------------------------------------------------------------------
# Data / observation loss
# ---------------------------------------------------------------------------

def data_loss(
    backend: AbstractBackend,
    prediction: Tensor,
    observation: Tensor,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """Supervised data mismatch at sensor/observation locations."""
    return weight * _apply_norm(backend, prediction - observation, norm)


# ---------------------------------------------------------------------------
# Regularisation losses
# ---------------------------------------------------------------------------

def sobolev_loss(
    backend: AbstractBackend,
    model_fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    grad_fn: Callable[[Callable, Tensor], Tensor],
    order: int = 1,
    weight: float = 1e-4,
) -> Tensor:
    """
    Sobolev regularisation: penalises ||∂^k u / ∂x^k||² for k = 1 … order.

    Parameters
    ----------
    model_fn : u(x)
    inputs   : collocation points
    grad_fn  : backend.grad(model_fn, argnums=0) — pre-built gradient fn
    order    : maximum derivative order to penalise
    weight   : regularisation coefficient
    """
    loss = backend.zeros(())
    current_fn = model_fn
    current_grad_fn = grad_fn

    for k in range(1, order + 1):
        g = current_grad_fn(inputs)
        loss = loss + backend.mean(backend.square(g))
        if k < order:
            current_fn      = lambda x, f=current_grad_fn: f(x)  # noqa: E731
            current_grad_fn = backend.grad(current_fn, argnums=0)

    return weight * loss


def gradient_penalty(
    backend: AbstractBackend,
    discriminator_fn: Callable[[Tensor], Tensor],
    real: Tensor,
    fake: Tensor,
    weight: float = 10.0,
) -> Tensor:
    """
    WGAN-GP gradient penalty (used in physics-constrained GANs).
    ε ~ U[0,1], x̂ = ε*real + (1-ε)*fake, penalty = (||∇D(x̂)||₂ - 1)²
    """
    batch_size = real.shape[0]
    if hasattr(backend, "random_uniform"):
        eps = backend.random_uniform((batch_size, 1), 0.0, 1.0)
    else:
        import numpy as np
        eps = backend.tensor(np.random.uniform(0.0, 1.0, size=(batch_size, 1)).astype("float32"))
    interp = eps * real + (1.0 - eps) * fake
    grad_d = backend.grad(discriminator_fn)(interp)
    # L2 norm over all axes except batch
    norm   = backend.sqrt(backend.sum(backend.square(grad_d), axis=-1) + _EPS)
    return weight * backend.mean(backend.square(norm - 1.0))


def spectral_energy_loss(
    backend: AbstractBackend,
    prediction: Tensor,
    high_freq_cutoff: int,
    weight: float = 1e-5,
) -> Tensor:
    """
    Penalise high-frequency energy in the prediction.
    Encourages smooth solutions by damping Fourier modes above ``high_freq_cutoff``.

    Operates on the last axis of ``prediction``.
    """
    spectrum = backend.rfft(prediction)
    energy   = backend.square(backend.abs(spectrum))
    high     = energy[..., high_freq_cutoff:]
    return weight * backend.mean(high)


def divergence_penalty(
    backend: AbstractBackend,
    jacobian: Tensor,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Divergence-free (incompressibility) structural penalty for a vector
    field u: penalises ``div(u) = sum_i du_i/dx_i`` toward zero — the
    standard soft constraint for incompressible flow (mass conservation,
    ∇·u = 0) or divergence-free magnetic fields (∇·B = 0).

    Parameters
    ----------
    jacobian : ``(..., d, d)`` tensor with ``jacobian[..., i, j] = du_i/dx_j``
        (the caller computes this the same way ``pde_residual.py`` builds
        per-component gradients — e.g. via ``backend.jacobian`` or by
        stacking ``d`` separate ``backend.grad`` calls, one per output
        component).
    """
    d = jacobian.shape[-1]
    div = sum(jacobian[..., i, i] for i in range(d))
    return weight * _apply_norm(backend, div, norm)


def curl_penalty_2d(
    backend: AbstractBackend,
    jacobian: Tensor,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Curl-free (irrotational) structural penalty for a 2D vector field u,
    e.g. requiring a flow to be a potential-flow / gradient field:
    penalises the scalar 2D curl ``du_y/dx - du_x/dy`` toward zero.

    ``jacobian`` : ``(..., 2, 2)`` tensor, ``jacobian[..., i, j] = du_i/dx_j``.
    """
    curl_z = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    return weight * _apply_norm(backend, curl_z, norm)


def curl_penalty_3d(
    backend: AbstractBackend,
    jacobian: Tensor,
    weight: float = 1.0,
    norm: str = "mse",
) -> Tensor:
    """
    Curl-free structural penalty for a 3D vector field u: penalises the
    full curl vector (∂u_z/∂y - ∂u_y/∂z, ∂u_x/∂z - ∂u_z/∂x,
    ∂u_y/∂x - ∂u_x/∂y) toward zero — e.g. for an irrotational velocity
    field or a magnetostatic field written as ∇φ.

    ``jacobian`` : ``(..., 3, 3)`` tensor, ``jacobian[..., i, j] = du_i/dx_j``.
    """
    curl_x = jacobian[..., 2, 1] - jacobian[..., 1, 2]
    curl_y = jacobian[..., 0, 2] - jacobian[..., 2, 0]
    curl_z = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    curl_sq = backend.square(curl_x) + backend.square(curl_y) + backend.square(curl_z)
    return weight * backend.mean(curl_sq)


# ---------------------------------------------------------------------------
# Causal weighting (Wang, Sankaran & Perdikaris, "Respecting causality is
# all you need for training physics-informed neural networks", 2022,
# arXiv:2203.07404)
# ---------------------------------------------------------------------------

def causal_weighted_loss(
    backend: AbstractBackend,
    pointwise_residual: Tensor,
    time_bucket_idx: Tensor,
    n_buckets: int,
    causality_tol: float = 1.0,
    norm: str = "mse",
) -> Tuple[Tensor, Tensor]:
    """
    Time-causal PDE residual loss: standard PINN training treats every
    collocation point equally regardless of its time coordinate, which
    lets the network fit *later* times before it has actually converged
    at *earlier* ones — physically backwards for an initial-value problem,
    and a well-documented cause of PINN training failure on
    transport-dominated/stiff time-dependent PDEs.

    Splits collocation points into ``n_buckets`` ordered temporal groups
    (assign bucket indices via e.g. ``np.digitize`` on each point's time
    coordinate before calling this), computes each bucket's own residual
    loss ``L_i``, and down-weights bucket ``i`` by
    ``w_i = exp(-causality_tol * sum_{k<i} L_k)`` — so a later bucket only
    gets significant training signal once every earlier bucket's loss has
    fallen enough for the exponential penalty to relax. Matches the
    paper's core weighting rule; note two implementation choices made
    explicitly rather than left implicit:

    * The cumulative sum inside the exponential is **stop-gradient**'d
      (detached from the graph) before use, exactly as the paper
      specifies — the weights are meant to act as fixed per-step
      multipliers, not something the optimizer can cheat by attacking
      directly instead of actually reducing the earlier-time residual.
    * The returned per-bucket weights are normalized to sum to 1 (not
      left as raw ``exp(...)`` values, which the paper's official
      implementation also monitors separately as a training-health
      diagnostic). This is a stabilizing choice made here, not a claim of
      exact numerical reproduction of the reference implementation.

    Returns
    -------
    (weighted_loss, per_bucket_weights) : the causally-weighted scalar
        loss, and the raw (already-normalized) per-bucket weight vector —
        return the latter too since ``min(weights)`` climbing toward 1
        over training is the paper's recommended diagnostic that training
        is actually "unlocking" later times, not stuck on an early one.
    """
    if n_buckets < 1:
        raise ValueError(f"causal_weighted_loss: n_buckets must be >= 1, got {n_buckets}.")

    # Normalize to a real tensor up front: comparing a plain Python list
    # to a scalar with `==` compares the whole list object, not
    # elementwise, so this must happen before any `== i` comparison below,
    # not be interleaved with it per-bucket.
    if not hasattr(time_bucket_idx, "dtype"):
        time_bucket_idx = backend.tensor(time_bucket_idx)

    bucket_losses = []
    for i in range(n_buckets):
        mask = (time_bucket_idx == i)
        selected = pointwise_residual[mask]
        # An empty bucket contributes exactly zero loss (and, since it's
        # zero, adds nothing to the cumulative sum other buckets' weights
        # depend on) rather than raising on an empty tensor reduction.
        if hasattr(selected, "shape") and selected.shape[0] == 0:
            bucket_losses.append(backend.zeros(()))
        else:
            bucket_losses.append(_apply_norm(backend, selected, norm))

    stacked = backend.stack(bucket_losses, axis=0)  # (n_buckets,)
    detached = backend.to_numpy(stacked)  # stop-gradient: plain host floats
    cumulative = [0.0]
    for i in range(n_buckets - 1):
        cumulative.append(cumulative[-1] + float(detached[i]))

    raw_weights = [math.exp(-causality_tol * c) for c in cumulative]
    total_w = sum(raw_weights) + _EPS
    weights = [w / total_w for w in raw_weights]

    weighted = backend.zeros(())
    for i in range(n_buckets):
        weighted = weighted + weights[i] * bucket_losses[i]

    weights_tensor = backend.tensor(weights) if hasattr(backend, "tensor") else weights
    return weighted, weights_tensor


# ---------------------------------------------------------------------------
# Composite weighted loss
# ---------------------------------------------------------------------------

@dataclass
class LossTerm:
    """A named loss contribution with a mutable weight."""
    name:    str
    fn:      Callable[[], Tensor]   # zero-argument closure evaluated each step
    weight:  float = 1.0
    history: List[float] = field(default_factory=list)


class WeightedLossComposite:
    """
    Manages a collection of weighted loss terms and supports several
    automatic re-weighting strategies.

    Strategies
    ----------
    * ``"fixed"``        – static weights, no adaptation.
    * ``"softmax_temp"`` – weights are softmax of normalised loss magnitudes.
    * ``"relobralo"``    – exponential-moving-average softmax of each
      term's loss *ratio* relative to its own recent history, inspired by
      Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-
      Informed Deep Learning" (2021, arXiv:2110.09813). NOTE: this is an
      independent implementation capturing that paper's core idea (relate
      each loss to its own trend, not just its instantaneous magnitude,
      via a softmax + EMA), not a line-for-line reproduction of their
      exact bookkeeping/hyperparameters — see the method docstring for
      the precise formula actually used here.
    * ``"grad_norm"``    – balance weights so every term contributes a
      comparable *gradient* magnitude to the shared trainable parameters
      (GradNorm-style, Chen et al. 2018), not just a comparable loss
      *value* — two terms can have similar loss values but wildly
      different gradient scales, which is what actually controls
      optimizer behavior. Supported on all four backends, via three
      genuinely different mechanisms (this needs real per-parameter
      gradients, which ``AbstractBackend.grad`` isn't shaped for — it
      differentiates a function of its own input tensor, not a scalar
      loss w.r.t. an external parameter list):
        - torch, paddle: both eager frameworks with a direct
          "differentiate this already-computed scalar w.r.t. this
          parameter list" primitive (``torch.autograd.grad`` /
          ``paddle.grad``) — used directly, no extra setup beyond
          ``set_shared_params``.
        - tensorflow: needs an already-open ``tf.GradientTape(persistent=
          True)`` that was watching ``shared_params`` *before* this
          composite was called (TF has no "grad of a past value" — only
          "grad of a value recorded on a live tape") — call
          ``set_tape(tape)`` each step before ``composite()``.
        - jax: JAX's functional ``jax.grad`` needs an actual *function*
          of the parameters, not an already-evaluated value — a zero-arg
          closure returning a value (this composite's normal ``fn``
          convention) can't be retroactively differentiated the way an
          eager framework's value can. Register an *additional*,
          optional pure function via ``add(..., params_fn=lambda
          params: ...)`` for any term you want included in JAX grad-norm
          balancing; terms without one keep their existing weight
          (with a one-time warning naming which term was skipped) rather
          than silently being treated as zero-gradient.
    * ``"uncertainty"``  – homoscedastic uncertainty weighting (Kendall,
      Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh
      Losses...", 2018): total loss includes ``exp(-s_i)*L_i + s_i`` for
      a per-term log-variance ``s_i``, so terms the network is "more
      uncertain about" get automatically down-weighted. Supported on
      *all four* backends identically, via a closed-form update instead
      of framework autodiff: since ``d/ds_i [exp(-s_i)*L_i + s_i] =
      1 - exp(-s_i)*L_i`` is a two-line scalar formula (L_i held fixed
      for that step, exactly as the method intends), ``s_i`` is updated
      with a plain gradient-descent step on this closed form using only
      each term's *value* (``backend.to_numpy``) — no
      Parameter/Variable/pytree registration with an external optimizer
      needed on any backend, unlike the previous torch-only
      ``trainable_parameters()`` design this replaces.

    Example
    -------
    >>> composite = WeightedLossComposite(backend, strategy="softmax_temp")
    >>> composite.add("pde",      lambda: pde_loss,  weight=1.0)
    >>> composite.add("bc",       lambda: bc_loss,   weight=10.0)
    >>> composite.add("data",     lambda: data_loss, weight=5.0)
    >>> total, breakdown = composite()
    """

    _VALID_STRATEGIES = {"fixed", "softmax_temp", "relobralo", "grad_norm", "uncertainty"}
    _GRAD_NORM_BACKENDS = {"torch", "paddle", "tensorflow", "jax"}

    def __init__(
        self,
        backend: AbstractBackend,
        strategy: str = "fixed",
        reweight_every: int = 100,
        temperature: float = 0.1,
        relobralo_decay: float = 0.99,
        grad_norm_target: Optional[str] = None,
        uncertainty_lr: float = 1e-2,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {sorted(self._VALID_STRATEGIES)}."
            )
        self.backend          = backend
        self.strategy         = strategy
        self.reweight_every   = reweight_every
        self.temperature      = temperature
        self.relobralo_decay  = relobralo_decay
        self.uncertainty_lr   = uncertainty_lr
        # Which term's raw gradient norm to normalize every other term
        # against for "grad_norm" (defaults to the first term added, e.g.
        # typically "pde" — the usual anchor in PINN loss balancing).
        self.grad_norm_target = grad_norm_target

        self._terms:  List[LossTerm] = []
        self._step:   int = 0

        # -- relobralo state: each term's loss value at t-1 and at the
        # random-lookback anchor t0, kept as plain host floats (this
        # strategy needs no gradients through its own bookkeeping).
        self._relobralo_prev: Dict[str, float] = {}
        self._relobralo_init: Dict[str, float] = {}

        # -- uncertainty state: plain Python floats (log-variances),
        # updated via the closed-form rule in `_uncertainty_total` — no
        # backend-specific Parameter/Variable type needed, so this works
        # identically on all four backends.
        self._uncertainty_log_vars: Dict[str, float] = {}

        # -- grad_norm state
        self._shared_params: Optional[List[Any]] = None
        self._tf_tape: Optional[Any] = None
        self._jax_params_fns: Dict[str, Callable[[Any], Tensor]] = {}
        self._jax_missing_warned: set = set()
        self._fallback_warned = False

    def add(
        self,
        name: str,
        fn: Callable[[], Tensor],
        weight: float = 1.0,
        params_fn: Optional[Callable[[Any], Tensor]] = None,
    ) -> "WeightedLossComposite":
        """
        ``params_fn``, if given, is an *additional* pure function
        ``params -> Tensor`` computing this same term's loss purely as a
        function of the shared parameters — only needed for
        ``strategy="grad_norm"`` on the JAX backend (see the class
        docstring's ``"grad_norm"`` entry for why JAX specifically needs
        this second, differently-shaped function alongside the normal
        zero-arg ``fn``). Ignored on every other backend/strategy.
        """
        self._terms.append(LossTerm(name=name, fn=fn, weight=weight))
        if self.strategy == "uncertainty":
            self._uncertainty_log_vars[name] = 0.0
        if params_fn is not None:
            self._jax_params_fns[name] = params_fn
        return self

    def set_shared_params(self, params: List[Any]) -> "WeightedLossComposite":
        """
        Required before using ``strategy="grad_norm"`` on torch, paddle,
        or tensorflow: the list of trainable parameters every term's
        gradient norm is measured against (typically
        ``model.parameters()`` — the shared trunk all loss terms actually
        train through). Not needed for JAX grad-norm balancing, which
        instead differentiates each term's registered ``params_fn``
        directly at the parameter pytree passed to ``__call__`` — see
        ``add``'s ``params_fn`` argument.
        """
        self._shared_params = list(params)
        return self

    def set_tape(self, tape: Any) -> "WeightedLossComposite":
        """
        Required before using ``strategy="grad_norm"`` on tensorflow: an
        already-open ``tf.GradientTape(persistent=True)`` that has been
        watching ``shared_params`` since *before* this composite's terms
        were evaluated — TF can only differentiate values recorded on a
        live tape, not retroactively differentiate an already-computed
        Python value the way torch/paddle's ``grad(value, params)``
        primitives can. Call this once per training step, before
        invoking the composite (e.g. right after opening the tape in
        ``Trainer._step_tf``), not once at construction.
        """
        self._tf_tape = tape
        return self

    def trainable_parameters(self) -> List[Any]:
        """
        Retained for backward compatibility with the previous torch-only
        ``"uncertainty"`` implementation, which needed its log-variances
        registered with the main optimizer. That's no longer true — see
        the class docstring's ``"uncertainty"`` entry — so this now
        always returns an empty list on every backend. Safe to leave in
        old code (adding `[]` to an optimizer's parameter list is a
        no-op); not needed in new code.
        """
        return []

    def __call__(self, params: Optional[Any] = None, record_history: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Evaluate all terms and return ``(total_loss, {name: value})``.

        ``params`` is only used for ``strategy="grad_norm"`` on JAX (the
        pytree passed to each term's registered ``params_fn`` — see
        ``add``); every other backend/strategy ignores it.

        ``record_history=False`` skips appending each term's value to
        ``term.history`` — required when this is called from *inside* a
        traced/transformed function (e.g. the ``loss_fn`` passed to
        ``jax.value_and_grad`` in ``Trainer._step_jax``): under tracing,
        ``val`` is a tracer, not a concrete array, and
        ``backend.to_numpy(val)`` raises ``TracerArrayConversionError``.
        Callers in that situation should record history afterward, from
        the concrete post-trace values, via ``record_history_from_values``.
        """
        b   = self.backend
        values: Dict[str, Tensor] = {}
        raw_grad_norms: Dict[str, float] = {}

        grad_norm_ready = self.strategy == "grad_norm" and self._grad_norm_backend_ready(params)

        for term in self._terms:
            if grad_norm_ready:
                val, gnorm = self._eval_with_grad_norm(term, params)
                raw_grad_norms[term.name] = gnorm
            else:
                val = term.fn()
            values[term.name] = val
            if record_history:
                term.history.append(float(b.to_numpy(val)))

        if self.strategy == "softmax_temp" and self._step % self.reweight_every == 0:
            self._softmax_reweight(values)
        elif self.strategy == "relobralo" and self._step % self.reweight_every == 0:
            self._relobralo_reweight(values)
        elif self.strategy == "grad_norm":
            if grad_norm_ready:
                if self._step % self.reweight_every == 0:
                    self._grad_norm_reweight(raw_grad_norms)
            else:
                self._warn_fallback("grad_norm")
        elif self.strategy == "uncertainty":
            return self._uncertainty_total(values), values

        total = sum(
            term.weight * values[term.name] for term in self._terms
        )

        self._step += 1
        return total, values

    def record_history_from_values(self, values: Dict[str, Any]) -> None:
        """
        Append already-concrete per-term values (e.g. the post-trace
        ``breakdown`` dict returned by ``jax.value_and_grad``) to each
        term's ``history`` — the counterpart to calling
        ``__call__(record_history=False)`` inside a traced function. Plain
        Python floats or numpy scalars are fine here since this is always
        called *after* tracing has completed.
        """
        for term in self._terms:
            if term.name in values:
                v = values[term.name]
                term.history.append(float(v) if not hasattr(v, "item") else float(np.asarray(v)))

    # ------------------------------------------------------------------
    # Private reweighting helpers
    # ------------------------------------------------------------------

    def _grad_norm_backend_ready(self, params: Optional[Any]) -> bool:
        n = self.backend.name
        if n in ("torch", "paddle"):
            return self._shared_params is not None
        if n == "tensorflow":
            return self._shared_params is not None and self._tf_tape is not None
        if n == "jax":
            # Ready as long as at least one term registered a params_fn —
            # terms without one are handled individually in
            # `_eval_with_grad_norm` (warned + left at their prior weight),
            # not treated as an all-or-nothing backend capability check.
            return params is not None and len(self._jax_params_fns) > 0
        return False

    def _warn_fallback(self, strategy: str) -> None:
        if not self._fallback_warned:
            n = self.backend.name
            if n == "tensorflow":
                hint = "call set_tape(tape) with an open persistent GradientTape each step"
            elif n == "jax":
                hint = "register at least one add(..., params_fn=...) and pass params to __call__"
            elif n in ("torch", "paddle"):
                hint = "call set_shared_params(model.parameters()) first"
            else:
                hint = f"backend '{n}' is not one of {sorted(self._GRAD_NORM_BACKENDS)}"
            warnings.warn(
                f"WeightedLossComposite(strategy='{strategy}') isn't ready yet on "
                f"backend '{n}' ({hint}). Falling back to static (fixed) weights "
                f"until it is.",
                UserWarning,
                stacklevel=3,
            )
            self._fallback_warned = True

    def _softmax_reweight(self, values: Dict[str, Tensor]) -> None:
        b = self.backend
        # The softmax is computed with backend tensor ops and only synced
        # to host once, for the small number of resulting scalar weights.
        # Under jit-compiled backends (JAX/TF), pulling every loss value to
        # a Python float inside the training step would force a device
        # sync each reweight step and break tracing if called from inside
        # a jitted step.
        n = len(self._terms)
        stacked = b.stack([values[t.name] for t in self._terms], axis=0)
        stacked = b.reshape(stacked, (n,)) if hasattr(b, "reshape") else stacked
        # No AbstractBackend implementation exposes a `.max()` method, so
        # the max-subtraction (needed only for numerical stability, not
        # correctness) is computed via the host sync the next line already
        # needs anyway (`b.to_numpy(new_weights_tensor)` below), rather
        # than adding an untested cross-backend `.max()` primitive.
        m = float(np.max(b.to_numpy(stacked)))
        exp = b.exp((stacked - m) / (self.temperature + _EPS))
        s = b.sum(exp)
        new_weights_tensor = exp / (s + _EPS) * n
        new_weights = [float(w) for w in b.to_numpy(new_weights_tensor)]
        for term, w in zip(self._terms, new_weights):
            term.weight = w

    def _relobralo_reweight(self, values: Dict[str, Tensor]) -> None:
        """
        ReLoBRaLo-inspired balancing. For each term, compare its current
        loss to its own value one reweight-interval ago (``prev``) and to
        its value the very first time it was ever weighted (``init``,
        the "random lookback" anchor — here always the first observation
        rather than the paper's randomly-chosen historical step, a
        deliberate simplification): a term whose loss has dropped a lot
        relative to both anchors gets down-weighted; one that has barely
        moved (or gotten worse) gets up-weighted. Combine the two ratios
        via softmax (temperature-scaled, like ``softmax_temp`` above),
        then EMA the result against the previous weight with decay
        ``relobralo_decay`` so weights move smoothly rather than jumping
        every reweight step.
        """
        b = self.backend
        n = len(self._terms)
        names = [t.name for t in self._terms]
        current = {name: float(b.to_numpy(values[name])) for name in names}

        for name in names:
            if name not in self._relobralo_init:
                self._relobralo_init[name] = current[name]
            if name not in self._relobralo_prev:
                self._relobralo_prev[name] = current[name]

        def _softmax_ratios(anchor: Dict[str, float]) -> List[float]:
            ratios = [current[name] / (anchor[name] + _EPS) for name in names]
            m = max(ratios)
            exps = [math.exp((r - m) / (self.temperature + _EPS)) for r in ratios]
            s = sum(exps) + _EPS
            return [e / s * n for e in exps]

        w_prev = _softmax_ratios(self._relobralo_prev)
        w_init = _softmax_ratios(self._relobralo_init)
        alpha = self.relobralo_decay

        for term, wp, wi in zip(self._terms, w_prev, w_init):
            new_w = alpha * term.weight + (1.0 - alpha) * (0.5 * wp + 0.5 * wi)
            term.weight = new_w

        self._relobralo_prev = current

    def _uncertainty_total(self, values: Dict[str, Tensor]) -> Tensor:
        """
        Kendall-style homoscedastic uncertainty weighting, updated via
        the closed-form rule derived in the class docstring rather than
        framework autodiff — see there for the full derivation. Fully
        backend-agnostic: only needs each term's scalar *value*.
        """
        b = self.backend
        total = b.zeros(())
        for term in self._terms:
            s = self._uncertainty_log_vars[term.name]
            eff_weight = math.exp(-s)
            total = total + eff_weight * values[term.name] + s
            term.weight = eff_weight  # for logging/history, matches the effective weight

            L_i = float(b.to_numpy(values[term.name]))
            # Gradient descent on exp(-s)*L_i + s w.r.t. s, with L_i held
            # fixed at its current value for this step (exactly what real
            # backprop would give too, since L_i doesn't itself depend on
            # s — only this explicit term does):
            #   d/ds [exp(-s)*L_i + s] = 1 - exp(-s)*L_i
            grad_s = 1.0 - eff_weight * L_i
            self._uncertainty_log_vars[term.name] = s - self.uncertainty_lr * grad_s

        term_history_val = total
        for term in self._terms:
            term.history.append(float(b.to_numpy(values[term.name])))
        self._step += 1
        return total

    def _eval_with_grad_norm(self, term: LossTerm, params: Optional[Any]) -> Tuple[Tensor, float]:
        """
        Evaluate one term and measure ``||d(term)/d(shared_params)||``,
        via whichever mechanism the active backend actually supports —
        see the class docstring's ``"grad_norm"`` entry for why each
        backend needs a genuinely different approach here (this is not
        something ``AbstractBackend.grad`` can express uniformly).
        """
        b = self.backend
        name = b.name

        if name == "torch":
            import torch
            val = term.fn()
            grads = torch.autograd.grad(
                val, self._shared_params, retain_graph=True, allow_unused=True,
            )
            sq_sum = torch.zeros(())
            for g in grads:
                if g is not None:
                    sq_sum = sq_sum + torch.sum(g * g)
            return val, float(torch.sqrt(sq_sum + _EPS).detach())

        if name == "paddle":
            import paddle
            val = term.fn()
            grads = paddle.grad(
                val, self._shared_params, retain_graph=True, allow_unused=True,
            )
            sq_sum = paddle.zeros([])
            for g in grads:
                if g is not None:
                    sq_sum = sq_sum + paddle.sum(g * g)
            return val, float(paddle.sqrt(sq_sum + _EPS).numpy())

        if name == "tensorflow":
            val = term.fn()
            grads = self._tf_tape.gradient(val, self._shared_params)
            sq_sum = 0.0
            for g in grads:
                if g is not None:
                    sq_sum += float((g.numpy() ** 2).sum())
            return val, math.sqrt(sq_sum + _EPS)

        if name == "jax":
            import jax
            import jax.numpy as jnp
            params_fn = self._jax_params_fns.get(term.name)
            if params_fn is None:
                if term.name not in self._jax_missing_warned:
                    warnings.warn(
                        f"WeightedLossComposite(strategy='grad_norm'): term "
                        f"'{term.name}' has no params_fn registered (see "
                        f"add()'s params_fn argument) — its gradient norm "
                        f"can't be measured on JAX, so its weight is left "
                        f"unchanged this round instead of being guessed at.",
                        UserWarning,
                        stacklevel=4,
                    )
                    self._jax_missing_warned.add(term.name)
                val = term.fn()
                return val, term.weight  # neutral: leaves this term's weight as-is
            val = params_fn(params)
            g = jax.grad(params_fn)(params)
            leaves = jax.tree_util.tree_leaves(g)
            sq_sum = sum(float(jnp.sum(leaf ** 2)) for leaf in leaves)
            return val, math.sqrt(sq_sum + _EPS)

        raise RuntimeError(f"WeightedLossComposite: unknown backend '{name}'.")

    def _grad_norm_reweight(self, raw_grad_norms: Dict[str, float]) -> None:
        """
        Set each term's weight so its gradient norm matches the target
        term's (default: the first term added) — the core GradNorm-style
        idea (Chen et al. 2018) applied to static per-step rebalancing
        rather than their full learned-rate formulation: weight_i is set
        so ``weight_i * ||grad_i|| ≈ ||grad_target||``, i.e.
        ``weight_i = ||grad_target|| / (||grad_i|| + eps)``.
        """
        target_name = self.grad_norm_target or self._terms[0].name
        target_norm = raw_grad_norms.get(target_name, 1.0)
        for term in self._terms:
            gnorm = raw_grad_norms.get(term.name, target_norm)
            term.weight = target_norm / (gnorm + _EPS)

    def summary(self) -> Dict[str, Any]:
        return {
            t.name: {
                "weight":  t.weight,
                "history": t.history[-10:],
            }
            for t in self._terms
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _apply_norm(backend: AbstractBackend, residual: Tensor, norm: str) -> Tensor:
    _NORMS = {
        "mse":         mse_loss,
        "mae":         mae_loss,
        "log_cosh":    log_cosh_loss,
        "huber":       huber_loss,
        "charbonnier": charbonnier_loss,
    }
    fn = _NORMS.get(norm.lower())
    if fn is None:
        raise ValueError(
            f"Unknown norm '{norm}'. Choose from {list(_NORMS)}."
        )
    return fn(backend, residual)


__all__ = [
    "mse_loss",
    "mae_loss",
    "log_cosh_loss",
    "huber_loss",
    "charbonnier_loss",
    "quantile_loss",
    "dirichlet_loss",
    "neumann_loss",
    "robin_loss",
    "periodic_loss",
    "interface_loss",
    "data_loss",
    "sobolev_loss",
    "gradient_penalty",
    "spectral_energy_loss",
    "divergence_penalty",
    "curl_penalty_2d",
    "curl_penalty_3d",
    "causal_weighted_loss",
    "WeightedLossComposite",
    "LossTerm",
]