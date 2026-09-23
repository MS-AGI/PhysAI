"""
physai/core/auto_optimizer.py

Automated configuration and heuristic brain for PhysAI training.

AutoOptimizer analyses the PDE problem geometry, model architecture,
and backend capabilities to emit a fully-specified RuntimeConfig that
the Trainer can consume directly.

Heuristic pipeline
------------------
1.  ``ProblemSpec``  – user declares PDE name, domain, loss weights, etc.
2.  ``AutoOptimizer.analyse()``   – runs heuristic rules
3.  ``RuntimeConfig``             – emitted configuration (frozen dataclass)
4.  ``Trainer``                   – consumes RuntimeConfig

Heuristics implemented
----------------------
* Learning-rate selection  (problem stiffness proxy via κ estimate)
* Optimizer choice         (L-BFGS for smooth, Adam for high-dim / chaotic)
* Collocation density      (dimension-aware, PDE-order-aware)
* Loss weighting           (residual-to-BC ratio from literature priors)
* Adaptive curriculum      (RAR – Residual-Adaptive Refinement schedule)
* Warm-up schedule         (cosine vs linear based on batch size)
* FNO vs PINN selection    (spectral content heuristic from domain size)
* Precision recommendation (float64 for stiff, float32 otherwise)
* Architecture sizing      (hidden width / depth from PDE complexity)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import tensorflow as tf
import torch

from physai.backends.base import AbstractBackend
from physai.geometry import Geometry

# ---------------------------------------------------------------------------
# PDE metadata registry
# ---------------------------------------------------------------------------
Tensor = Union["torch.Tensor", "tf.Tensor", "jax.Array"]

@dataclass(frozen=True)
class PDEMeta:
    """Static metadata about a PDE class, used by heuristics."""
    name:           str
    order:          int     # highest derivative order
    nonlinear:      bool    # is the PDE nonlinear?
    n_components:   int     # number of coupled output fields
    stiff:          bool    # known to be stiff (small ε, large wave numbers)
    spectral_bias:  str     # "low" | "mixed" | "high"  – expected frequency content
    recommended_arch: str   # "pinn" | "fno" | "both"

_PDE_META: Dict[str, PDEMeta] = {
    # =========================================================================
    # ORIGINAL METADATA ENTRIES
    # =========================================================================
    "poisson":              PDEMeta("poisson",        2, False, 1, False, "low",   "pinn"),
    "laplace":              PDEMeta("laplace",        2, False, 1, False, "low",   "pinn"),
    "heat":                 PDEMeta("heat",           2, False, 1, True,  "low",   "pinn"),
    "diffusion":            PDEMeta("diffusion",      2, False, 1, True,  "low",   "pinn"),
    "wave":                 PDEMeta("wave",           2, False, 1, False, "mixed", "fno"),
    "burgers":              PDEMeta("burgers",        2, True,  1, True,  "high",  "fno"),
    "navier_stokes":        PDEMeta("navier_stokes",  2, True,  3, True,  "high",  "fno"),
    "advection":            PDEMeta("advection",      1, False, 1, False, "high",  "fno"),
    "helmholtz":            PDEMeta("helmholtz",      2, False, 1, False, "mixed", "both"),
    "allen_cahn":           PDEMeta("allen_cahn",     2, True,  1, True,  "mixed", "pinn"),
    "cahn_hilliard":        PDEMeta("cahn_hilliard",  4, True,  2, True,  "mixed", "pinn"),
    "schrodinger":          PDEMeta("schrodinger",    2, False, 2, True,  "mixed", "fno"),
    "klein_gordon":         PDEMeta("klein_gordon",   2, False, 1, False, "mixed", "fno"),
    "reaction_diffusion":   PDEMeta("reaction_diffusion", 2, True, 1, True, "mixed", "pinn"),
    "eikonal":              PDEMeta("eikonal",        1, False, 1, False, "low",   "pinn"),
    "darcy":                PDEMeta("darcy",          2, False, 1, False, "low",   "fno"),
    "euler":                PDEMeta("euler",          1, True,  3, True,  "high",  "fno"),
    "biharmonic":           PDEMeta("biharmonic",     4, False, 1, True,  "low",   "pinn"),
    "stokes":               PDEMeta("stokes",         2, False, 3, False, "low",   "pinn"),
    "nls":                  PDEMeta("nls",            2, True,  2, True,  "mixed", "fno"),
    "kdv":                  PDEMeta("kdv",            3, True,  1, True,  "high",  "fno"),
    "fokker_planck":        PDEMeta("fokker_planck",  2, True,  1, False, "mixed", "pinn"),

    # =========================================================================
    # NEW SUITE: ADVANCED RESEARCH & INDUSTRIAL SYSTEMS METADATA
    # =========================================================================
    "nlse_2d":              PDEMeta("nlse_2d",              2, True,  2, True,  "mixed", "both"),
    "sine_gordon_2d":       PDEMeta("sine_gordon_2d",       2, True,  1, True,  "mixed", "fno"),
    "fisher_kpp":           PDEMeta("fisher_kpp",           2, True,  1, True,  "low",   "pinn"),
    "fitzhugh_nagumo":      PDEMeta("fitzhugh_nagumo",      2, True,  2, True,  "mixed", "pinn"),
    "kuramoto_sivashinsky": PDEMeta("kuramoto_sivashinsky", 4, True,  1, True,  "high",  "both"),
    "kadomtsev_petviashvili": PDEMeta("kadomtsev_petviashvili", 4, True, 1, True,  "high",  "fno"),
    "porous_medium":        PDEMeta("porous_medium",        2, True,  1, True,  "mixed", "pinn"),
    "biharmonic_steady":    PDEMeta("biharmonic_steady",    4, False, 1, False, "low",   "pinn"),
    "boussinesq_wave":      PDEMeta("boussinesq_wave",      4, True,  1, True,  "high",  "fno"),
    "euler_tricomi":        PDEMeta("euler_tricomi",        2, False, 1, False, "high",  "pinn"),
    "drift_diffusion_poisson": PDEMeta("drift_diffusion_poisson", 2, True, 2, True, "high", "pinn"),
    "shallow_water_2d":     PDEMeta("shallow_water_2d",     1, True,  3, True,  "high",  "fno"),
    "heston_volatility":    PDEMeta("heston_volatility",    2, True,  1, True,  "mixed", "both"),
    "phase_field_crystal":  PDEMeta("phase_field_crystal",  4, True,  1, True,  "high",  "pinn"),
    "brinkman_darcy":       PDEMeta("brinkman_darcy",       2, False, 3, False, "low",   "pinn"),
    "perona_malik":         PDEMeta("perona_malik",         2, True,  1, True,  "mixed", "fno"),
    "burgers_2d":           PDEMeta("burgers_2d",           2, True,  2, True,  "high",  "fno"),
    "boussinesq_convection": PDEMeta("boussinesq_convection", 2, True, 4, True, "high",  "both"),
    "dendritic_solidification": PDEMeta("dendritic_solidification", 2, True, 2, True, "mixed", "pinn"),
    "relativistic_fluid":   PDEMeta("relativistic_fluid",   1, True,  2, True,  "high",  "fno"),

    # =========================================================================
    # PATCH: 11 PDE_REGISTRY entries that had no PDEMeta and were silently
    # falling back to AutoOptimizer.analyse()'s generic defaults (order=2,
    # nonlinear=False, n_components=1, ...) — wrong for e.g. the 2-component
    # systems below, which would have broken output-slicing downstream.
    # Values below are read directly off each residual class's own
    # docstring/__call__ (order = highest derivative order in the strong
    # form; n_components = number of output fields model_fn must return);
    # stiff/spectral_bias/recommended_arch are set to match the closest
    # sibling PDE already present in this table.
    # =========================================================================
    "klein_gordon_nonlinear_2d": PDEMeta("klein_gordon_nonlinear_2d", 2, True, 1, True, "mixed", "fno"),
    "cahn_hilliard_2d":       PDEMeta("cahn_hilliard_2d",       4, True,  2, True,  "mixed", "pinn"),
    "fokker_planck_2d":       PDEMeta("fokker_planck_2d",       2, False, 1, False, "mixed", "pinn"),
    "swift_hohenberg":        PDEMeta("swift_hohenberg",        4, True,  1, True,  "high",  "both"),
    "black_scholes_2d":       PDEMeta("black_scholes_2d",       2, False, 1, False, "low",   "pinn"),
    "regularised_long_wave":  PDEMeta("regularised_long_wave",  3, True,  1, True,  "high",  "fno"),
    "gierer_meinhardt":       PDEMeta("gierer_meinhardt",       2, True,  2, True,  "mixed", "pinn"),
    "gray_scott":             PDEMeta("gray_scott",             2, True,  2, True,  "mixed", "pinn"),
    "viscous_wave":           PDEMeta("viscous_wave",           3, False, 1, True,  "mixed", "fno"),
    "radhakrishnan_kundu_lakshmanan": PDEMeta("radhakrishnan_kundu_lakshmanan", 3, True, 2, True, "high", "fno"),
    "complex_ginzburg_landau_2d": PDEMeta("complex_ginzburg_landau_2d", 2, True, 2, True, "mixed", "both"),

    # =========================================================================
    # Einstein Field Equations (see core/pde_residual.py::EinsteinFieldResidual)
    # order=2: Riemann/Ricci are built from second derivatives of g_munu.
    # n_components=10: independent components of the symmetric metric
    # g_munu that model_fn must output (and the residual returns).
    # nonlinear=True: Einstein tensor is quadratic-and-worse in Gamma, which
    # is itself built from g^-1 (already nonlinear) and dg.
    # stiff=True / spectral_bias="high": nested double-backward over a
    # 10 -> 64 -> 10 component pipeline is by far the most ill-conditioned,
    # highest-order-effective residual in this table; treat it like the
    # other 4th-order / tensor-coupled entries above, not like a scalar PDE.
    # recommended_arch="pinn": curvature fields over irregular/physically
    # meaningful domains (not periodic grids), same reasoning as stokes/
    # biharmonic/brinkman_darcy above — FNO's grid-spectral assumption
    # doesn't fit this any better than it fits those.
    # =========================================================================
    "einstein_field":        PDEMeta("einstein_field",        2, True, 10, True, "high", "pinn"),
    "efe":                   PDEMeta("efe",                   2, True, 10, True, "high", "pinn"),

    # =========================================================================
    # Sprint 2: Phonons, Dirac, Quantum Gases (Bose-Einstein / Fermi-Dirac),
    # Quantum Relativistic Fluid. See core/pde_residual.py for each class.
    # =========================================================================
    # Phonon: 4th-order dispersive correction on top of the base wave eq.,
    # same order as the other lattice/dispersive entries above (kdv,
    # boussinesq_wave); single scalar field, non-stiff at normal wavenumbers.
    "phonon":                PDEMeta("phonon",                4, True,  1, True,  "high",  "fno"),
    "phonons":               PDEMeta("phonons",                4, True,  1, True,  "high",  "fno"),
    # Dirac (1+1-D): first-order hyperbolic system, 4 real components
    # (2-component complex spinor), linear (no self-interaction term) but
    # numerically stiff for small mass (near-massless dispersion).
    "dirac":                 PDEMeta("dirac",                 1, False, 4, True,  "mixed", "pinn"),
    "dirac_equation":        PDEMeta("dirac_equation",        1, False, 4, True,  "mixed", "pinn"),
    # Bose-Einstein / Gross-Pitaevskii: same order/stiffness/spectral
    # profile as the existing "nls" entry (GPE *is* NLS + a trap term),
    # 2 real components (real/imag of psi).
    "bose_einstein":         PDEMeta("bose_einstein",         2, True,  2, True,  "mixed", "fno"),
    "gross_pitaevskii":      PDEMeta("gross_pitaevskii",      2, True,  2, True,  "mixed", "fno"),
    # Fermi gas: Euler-type first-order system (n, u) with a stiff n^(5/3)
    # closure at low density (steep pressure gradient) — same structure as
    # "euler" above but with the degenerate-gas equation of state.
    "fermi_gas":              PDEMeta("fermi_gas",              1, True,  2, True,  "high",  "fno"),
    "fermi_dirac":            PDEMeta("fermi_dirac",            1, True,  2, True,  "high",  "fno"),
    # Quantum relativistic fluid: relativistic_fluid's energy/momentum
    # system plus a Bohm quantum-potential term that adds a genuine 2nd
    # spatial derivative -> bump order 1 -> 2 relative to that entry.
    "quantum_relativistic_fluid": PDEMeta("quantum_relativistic_fluid", 2, True, 2, True, "high", "fno"),
}


# ---------------------------------------------------------------------------
# Problem specification (user-facing)
# ---------------------------------------------------------------------------

@dataclass
class DomainSpec:
    """
    Spatial-temporal domain description.

    ``bounds`` is always required — it is the axis-aligned bounding box
    used by every heuristic in this file (stiffness proxy, collocation
    density, spectral-content check, etc.) regardless of the true shape
    of the domain, and by ``_uniform_sample``'s fallback path.

    ``geometry`` is optional. When given, it must be a ``physai.geometry.
    Geometry`` instance (box, CSG combination, custom SDF, or a mesh
    loaded via ``geometry_from_file``) whose actual shape can be
    *contained inside* ``bounds`` but need not fill it — e.g. a torus,
    an L-shaped domain, or an uploaded STL part. When set, sampling
    (``_uniform_sample``, RAR candidate pools, and ``Trainer``'s
    boundary-condition sampling) draws from the true geometry via
    rejection sampling instead of treating the domain as a solid box.
    If omitted, the domain is exactly the hyper-rectangle in ``bounds``,
    matching the library's original behaviour.
    """
    spatial_dims:  int                          # number of spatial dimensions
    bounds:        List[Tuple[float, float]]    # [(x_min, x_max), ...] — bounding box
    time_domain:   Optional[Tuple[float, float]] = None  # (t0, T) or None (steady)
    periodic_axes: List[int] = field(default_factory=list)
    geometry:      Optional[Geometry] = None     # arbitrary shape inside `bounds`

    def __post_init__(self) -> None:
        if self.geometry is not None:
            if self.geometry.dim != self.spatial_dims:
                raise ValueError(
                    f"DomainSpec.geometry has dim={self.geometry.dim} but "
                    f"spatial_dims={self.spatial_dims} — they must match."
                )
            # Bounds must contain the geometry's own bounding box; a
            # mismatch here silently truncates the shape during rejection
            # sampling, so it's worth catching immediately rather than
            # producing a training set with a bite taken out of it.
            for i, (lo, hi) in enumerate(self.geometry.bounds):
                blo, bhi = self.bounds[i]
                if lo < blo - 1e-9 or hi > bhi + 1e-9:
                    raise ValueError(
                        f"DomainSpec.geometry's bounding box on axis {i} is "
                        f"[{lo}, {hi}], which is not contained in "
                        f"DomainSpec.bounds[{i}] = ({blo}, {bhi}). Widen "
                        f"`bounds` (or tighten the geometry) so it fully "
                        f"encloses the shape."
                    )


@dataclass
class ProblemSpec:
    """
    Complete problem specification provided by the user.

    Parameters
    ----------
    pde_name       : one of the keys in PDE_REGISTRY
    domain         : DomainSpec
    n_collocation  : override for number of collocation points (None → auto)
    n_bc_points    : override for number of boundary points   (None → auto)
    model_arch     : "pinn" | "fno" | "auto"
    backend_name   : "torch" | "jax" | "tensorflow"
    target_loss    : training convergence threshold
    max_epochs     : hard stop
    use_lbfgs_phase: whether to run L-BFGS fine-tuning after Adam warm-up
    extra_params   : any PDE-specific parameters forwarded to residual
    """
    pde_name:         str
    domain:           DomainSpec
    n_collocation:    Optional[int]  = None
    n_bc_points:      Optional[int]  = None
    model_arch:       str            = "auto"
    backend_name:     str            = "torch"
    target_loss:      float          = 1e-5
    max_epochs:       int            = 20_000
    use_lbfgs_phase:  bool           = True
    extra_params:     Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime configuration (emitted by AutoOptimizer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    name:       str             # "cosine" | "step" | "exponential" | "none"
    warmup_steps: int = 0
    decay_steps:  int = 10_000
    min_lr:       float = 1e-6
    gamma:        float = 0.95  # for step/exponential
    step_size:    int   = 1_000


@dataclass(frozen=True)
class ModelConfig:
    arch:             str             # "pinn" | "fno" | "spectral_element"
    layer_sizes:      Tuple[int, ...]
    activation:       str
    use_residual:     bool
    # FNO-specific
    fno_modes:        Optional[Tuple[int, ...]] = None
    fno_width:        int                       = 64
    fno_n_layers:     int                       = 4
    fno_projection_channels: int                = 128
    # spectral_element-specific (USENOCPModule / build_spectral_element_trainer)
    n_elements:       Optional[int]             = None
    n_modes:          Optional[int]             = None
    rank:             Optional[int]             = None


@dataclass(frozen=True)
class TrainingConfig:
    optimizer_name:     str
    learning_rate:      float
    weight_decay:       float
    batch_size:         int
    n_collocation:      int
    n_bc_points:        int
    loss_weights:       Dict[str, float]
    scheduler:          SchedulerConfig
    use_lbfgs_phase:    bool
    lbfgs_max_iter:     int
    rar_enabled:        bool
    rar_interval:       int
    rar_fraction:       float        # fraction of collocation pts to re-sample
    dtype:              str          # "float32" | "float64"
    grad_clip_norm:     Optional[float]
    use_jit:            bool


@dataclass(frozen=True)
class RuntimeConfig:
    """Complete, frozen configuration emitted by AutoOptimizer."""
    problem:  ProblemSpec
    model:    ModelConfig
    training: TrainingConfig
    meta:     PDEMeta

    def as_dict(self) -> Dict[str, Any]:
        return {
            "problem":  asdict(self.problem),
            "model":    asdict(self.model),
            "training": asdict(self.training),
            "meta":     asdict(self.meta),
        }


# ---------------------------------------------------------------------------
# Core heuristic functions
# ---------------------------------------------------------------------------

def _estimate_stiffness_proxy(meta: PDEMeta, domain: DomainSpec) -> float:
    """
    Rough stiffness proxy κ ∈ [0, 1] based on PDE order, nonlinearity,
    and domain aspect ratio.
    Higher κ → stiffer problem → prefer smaller LR, float64, L-BFGS.
    """
    kappa = 0.0
    kappa += 0.25 * min(meta.order / 4.0, 1.0)
    kappa += 0.20 if meta.nonlinear else 0.0
    kappa += 0.20 if meta.stiff     else 0.0

    # Domain size penalty: large domains need more modes
    sizes = [abs(b - a) for a, b in domain.bounds]
    aspect = max(sizes) / (min(sizes) + 1e-8)
    kappa += 0.10 * min(aspect / 10.0, 1.0)

    # Time horizon penalty
    if domain.time_domain is not None:
        T = domain.time_domain[1] - domain.time_domain[0]
        kappa += 0.10 * min(T / 10.0, 1.0)

    return min(kappa, 1.0)


def _select_optimizer(
    kappa: float,
    meta: PDEMeta,
    n_dims: int,
    use_lbfgs_phase: bool,
) -> Tuple[str, float, float]:
    """
    Returns (optimizer_name, lr, weight_decay) matching the problem profile.
    Guarantees parameter safety for backward execution engines.
    """
    # 1. Core Heuristic Selection Flow
    # The initial optimizer choice varies with the problem profile; L-BFGS
    # handoff is orchestrated separately via use_lbfgs_phase in analyse().
    # AdamW is used whenever weight decay is warranted.
    if n_dims >= 4 or meta.nonlinear:
        # High-dimensional or chaotic: Adam is a safe default baseline
        lr = max(1e-4, 5e-4 * (1.0 - 0.5 * kappa))
        opt_name = "adam"
        weight_decay = 0.0
    elif kappa > 0.6 and use_lbfgs_phase:
        # Stiff but smooth: start with AdamW (decay helps generalisation
        # during warm-up), hand off to L-BFGS in the trainer for fine-tuning
        lr = max(5e-5, 1e-3 * (1.0 - kappa))
        opt_name = "adamw"
        weight_decay = 1e-6
    else:
        # Low stiffness, smooth, low-dimensional: hand the whole run to
        # the dedicated L-BFGS fine-tuning phase (Trainer._lbfgs_phase)
        # rather than building torch.optim.LBFGS as the *main*-loop
        # optimizer. torch.optim.LBFGS.step() requires a `closure` that
        # recomputes loss/backward; Trainer's main loop
        # (backend.optimizer_step -> optimizer.step()) never supplies
        # one, so opt_name must stay "adam" for the main loop, with the
        # L-BFGS *phase* forced on instead — this is also what
        # "kappa > 0.6 and use_lbfgs_phase" already does one branch up,
        # so this just extends the same handoff pattern to the
        # low-kappa case instead of trying to run LBFGS standalone.
        lr = max(1e-4, 1e-3 * (1.0 - 0.4 * kappa))
        opt_name = "adam"
        weight_decay = 0.0

    # 2. Dynamic Compatibility Guard Matrix
    # Only optimizers that genuinely error/no-op on weight_decay go here.
    # "adam" and "lbfgs" have no native decoupled decay; "adamw" does.
    DECAY_UNSUPPORTED = {"adam", "sgd", "rmsprop", "adagrad", "lbfgs"}

    if opt_name in DECAY_UNSUPPORTED and weight_decay > 0.0:
        # Automatically degrade weight decay to zero to safeguard backends (like Optax)
        weight_decay = 0.0

    return opt_name, lr, weight_decay

def _select_collocation(
    meta: PDEMeta,
    domain: DomainSpec,
    n_override: Optional[int],
) -> int:
    if n_override is not None:
        return n_override

    d     = domain.spatial_dims + (1 if domain.time_domain else 0)
    base  = 500 * (2 ** min(d - 1, 4))   # exponential growth with dimension
    order_mult = 1.5 ** (meta.order - 1)  # higher-order PDEs need more pts
    nl_mult    = 1.5 if meta.nonlinear else 1.0
    return int(base * order_mult * nl_mult)


def _select_bc_points(
    domain: DomainSpec,
    n_override: Optional[int],
    n_collocation: int,
) -> int:
    if n_override is not None:
        return n_override
    # Typically 10–20% of interior points
    return max(100, n_collocation // 8)


def _select_model_arch(
    spec: ProblemSpec,
    meta: PDEMeta,
    backend: AbstractBackend,
) -> str:
    if spec.model_arch != "auto":
        return spec.model_arch.lower()

    # Prefer FNO for high-frequency, nonlinear, or time-dependent problems
    if meta.recommended_arch == "fno":
        return "fno"
    if meta.recommended_arch == "pinn":
        return "pinn"
    if meta.recommended_arch == "spectral_element":
        # USENOCPModule's backbone takes a scalar time input (spatial
        # structure lives in the CP-factored Chebyshev coefficients, not
        # the network's input) — see build_spectral_element_trainer's
        # docstring. It only makes sense for time-dependent problems, so
        # fall back to the "both" dimension-based heuristic otherwise
        # rather than emitting an arch that build_spectral_element_trainer
        # will immediately reject.
        if spec.domain.time_domain is not None:
            return "spectral_element"
        return "fno" if spec.domain.spatial_dims >= 2 else "pinn"
    # "both" → decide by dimension
    return "fno" if spec.domain.spatial_dims >= 2 else "pinn"


def _select_activation(meta: PDEMeta) -> str:
    """
    Pick an activation, preferring the PINN-specific ones (added to all
    three backends specifically for this) over generic ML activations
    wherever they're the better-motivated choice for the residual class:

    - "llaaf"  Layer-wise Locally Adaptive tanh (Jagtap & Karniadakis
               2020): a*tanh(n*a*x), `a` learnable per layer. Lets the
               network locally sharpen/flatten its own nonlinearity where
               the residual's gradient is steep -- documented PINN
               convergence-speed gains over static activations, and
               unlike gelu/silu it doesn't flatten (with vanishing higher
               derivatives) in saturated regions, so it's safe even for
               high-order residuals.
    - "snake"  x + sin(x)^2 (Ziyin et al. 2020): periodic-plus-linear,
               good for convection-dominated / shock-forming residuals
               without SIREN's brittle initialization requirements.
    - "pade"   Learnable rational P(x)/Q(x) (Molina et al. 2019, safe
               variant): smooth, spectral-bias-resistant, most expressive
               of the three but 6 extra learnable scalars per layer.
    - "tanh"   Classic PINN baseline where none of the above have a
               specific edge (plain linear, non-stiff, low-order).
    """
    # 1. High-order operators (Cahn-Hilliard, biharmonic,
    #    Kuramoto-Sivashinsky, KdV family, Swift-Hohenberg,
    #    phase-field-crystal, ...): derivative fidelity across many
    #    autodiff passes dominates. L-LAAF's per-layer adaptive slope is
    #    built for exactly this; plain tanh is the safe fallback these
    #    used before L-LAAF existed here, so use the strictly better
    #    option now that it's available.
    if meta.order >= 3:
        return "llaaf"

    # 2. Convection-dominated / shock-forming, order <= 2 (Burgers,
    #    Navier-Stokes, Euler, advection, relativistic fluid, ...):
    #    Snake's periodic-plus-linear shape captures the sharp,
    #    near-discontinuous gradients in these residuals better than a
    #    monotonic squashing activation, without needing SIREN-style
    #    initialization changes elsewhere in the model builder.
    if meta.spectral_bias == "high":
        return "snake"

    # 3. Moderately nonlinear reaction/diffusion-type systems (Allen-Cahn,
    #    Gray-Scott, FitzHugh-Nagumo, Fisher-KPP, ...): the safe Pade unit
    #    gives more expressive per-layer curvature than a fixed gelu/silu
    #    shape, at modest extra parameter cost, and has no poles by
    #    construction so it's safe to turn on by default here.
    if meta.nonlinear:
        return "pade"

    # 4. Linear but stiff (heat/diffusion-like): elu's saturating negative
    #    branch handles stiffness-driven activation saturation a bit more
    #    gracefully than plain tanh.
    if meta.stiff:
        return "elu"

    # 5. Linear, non-stiff, low/mixed spectral content (Poisson, Helmholtz,
    #    Stokes, Darcy, Eikonal, ...): tanh, the classic elliptic baseline.
    return "tanh"


def _build_model_config(
    arch: str,
    meta: PDEMeta,
    domain: DomainSpec,
    kappa: float,
) -> ModelConfig:
    d          = domain.spatial_dims + (1 if domain.time_domain else 0)
    n_in       = d
    n_out      = meta.n_components

    # Network width scales with stiffness and dimensionality
    base_width = 64
    width      = int(base_width * (1 + kappa) * math.sqrt(d))
    width      = min(max(width, 32), 512)

    # Depth: higher-order PDEs benefit from deeper networks
    depth = 3 + meta.order
    depth = min(depth, 8)

    layer_sizes = (n_in,) + (width,) * depth + (n_out,)

    activation   = _select_activation(meta)
    use_residual = depth >= 5

    if arch == "fno":
        # Fourier mode count: domain-size dependent
        sizes = [abs(b - a) for a, b in domain.bounds]
        modes = tuple(
            int(min(16 * s / (max(sizes) + 1e-8) + 4, 32))
            for s in sizes
        )
        if domain.time_domain:
            modes = modes + (12,)  # time modes
        fno_width = width
        return ModelConfig(
            arch=arch,
            layer_sizes=layer_sizes,
            activation=activation,
            use_residual=use_residual,
            fno_modes=modes,
            fno_width=fno_width,
            fno_n_layers=max(3, depth - 1),
            fno_projection_channels=fno_width * 2,
        )

    if arch == "spectral_element":
        # USENOCPModule sizing: n_elements partitions the time axis (one
        # spectral element per training-time subdomain), n_modes is the
        # per-element Chebyshev truncation, rank is the CP-decomposition
        # rank used to keep the per-element tensor from scaling
        # exponentially with `dims`. Same stiffness/dimension-aware spirit
        # as the PINN width/depth heuristics above, just for this arch's
        # own parameters — build_spectral_element_trainer still requires
        # the caller's own dim_coeffs (the PDE's linear operator), since
        # that's genuinely problem-specific and can't be inferred here.
        n_elements = max(2, int(2 + 4 * kappa))
        n_modes    = min(max(8, int(8 * (1 + kappa))), 32)
        rank       = min(max(4, n_out * 2), 16)
        return ModelConfig(
            arch=arch,
            layer_sizes=layer_sizes,
            activation=activation,
            use_residual=use_residual,
            n_elements=n_elements,
            n_modes=n_modes,
            rank=rank,
        )

    return ModelConfig(
        arch=arch,
        layer_sizes=layer_sizes,
        activation=activation,
        use_residual=use_residual,
    )


def _build_loss_weights(meta: PDEMeta, kappa: float) -> Dict[str, float]:
    """
    Prior-informed loss weights from literature.

    PDE residual weight is 1.0 (reference). BC and IC weights are
    elevated for stiff, nonlinear, or high-order problems.
    """
    bc_factor = 10.0 * (1.0 + kappa)
    ic_factor = 5.0  * (1.0 + kappa) if meta.stiff else 1.0
    return {
        "pde":  1.0,
        "bc":   bc_factor,
        "ic":   ic_factor,
        "data": 1.0,
    }


def _build_scheduler(
    optimizer_name: str,
    max_epochs: int,
    n_collocation: int,
    kappa: float,
) -> SchedulerConfig:
    warmup = max(200, max_epochs // 50)
    if optimizer_name == "adam":
        return SchedulerConfig(
            name        = "cosine",
            warmup_steps = warmup,
            decay_steps  = max_epochs,
            min_lr       = 1e-7,
        )
    return SchedulerConfig(name="none")


def _select_dtype(meta: PDEMeta, kappa: float, backend_name: str) -> str:
    # JAX defaults to float32 unless x64 enabled; TF float32 is usually fine
    if backend_name == "jax":
        return "float64" if (meta.stiff and kappa > 0.7) else "float32"
    return "float64" if (meta.stiff and kappa > 0.8) else "float32"


# ---------------------------------------------------------------------------
# AutoOptimizer
# ---------------------------------------------------------------------------

class AutoOptimizer:
    """
    Analyses a ProblemSpec and emits a RuntimeConfig.

    Parameters
    ----------
    backend : AbstractBackend instance (used to query capabilities)
    verbose : if True, print heuristic reasoning to stdout
    """

    def __init__(
        self,
        backend: AbstractBackend,
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.verbose = verbose

    def analyse(self, spec: ProblemSpec) -> RuntimeConfig:
        """
        Run the full heuristic pipeline and return a RuntimeConfig.

        Parameters
        ----------
        spec : ProblemSpec

        Returns
        -------
        RuntimeConfig (frozen)
        """
        pde_key = spec.pde_name.lower()
        if pde_key not in _PDE_META:
            warnings.warn(
                f"PDE '{spec.pde_name}' not in metadata registry. "
                "Using generic defaults.",
                UserWarning,
            )
            meta = PDEMeta(
                name=pde_key, order=2, nonlinear=False,
                n_components=1, stiff=False,
                spectral_bias="mixed", recommended_arch="pinn",
            )
        else:
            meta = _PDE_META[pde_key]

        # _select_optimizer and _select_collocation (below) both use the
        # same full dimensionality convention: spatial dims plus the time
        # dimension when the problem is time-dependent.
        d  = spec.domain.spatial_dims + (1 if spec.domain.time_domain else 0)
        kappa = _estimate_stiffness_proxy(meta, spec.domain)

        opt_name, lr, wd = _select_optimizer(
            kappa, meta, d, spec.use_lbfgs_phase
        )
        n_coll  = _select_collocation(meta, spec.domain, spec.n_collocation)
        n_bc    = _select_bc_points(spec.domain, spec.n_bc_points, n_coll)
        arch    = _select_model_arch(spec, meta, self.backend)
        m_cfg   = _build_model_config(arch, meta, spec.domain, kappa)
        weights = _build_loss_weights(meta, kappa)
        sched   = _build_scheduler(opt_name, spec.max_epochs, n_coll, kappa)
        dtype   = _select_dtype(meta, kappa, spec.backend_name)

        # RAR: activate for nonlinear or high-stiffness problems
        rar_enabled  = meta.nonlinear or kappa > 0.5
        rar_interval = max(100, spec.max_epochs // 40)
        rar_fraction = 0.1 + 0.1 * kappa

        # Gradient clipping: use for high-stiffness
        grad_clip = 1.0 if kappa > 0.6 else None

        # JIT: use when backend supports it
        use_jit = self.backend.capabilities.supports_jit

        # The low-kappa branch of _select_optimizer now always routes
        # through Adam-main-loop + L-BFGS-fine-tune (see its comment) —
        # so use_lbfgs_phase must trigger there regardless of kappa, not
        # only above the 0.4 threshold, or that problem class never gets
        # any L-BFGS refinement at all.
        force_lbfgs_phase = opt_name == "adam" and kappa <= 0.6
        use_lbfgs_phase   = spec.use_lbfgs_phase and (kappa > 0.4 or force_lbfgs_phase)

        t_cfg = TrainingConfig(
            optimizer_name   = opt_name,
            learning_rate    = lr,
            weight_decay     = wd,
            batch_size       = min(n_coll, 4096),
            n_collocation    = n_coll,
            n_bc_points      = n_bc,
            loss_weights     = weights,
            scheduler        = sched,
            use_lbfgs_phase  = use_lbfgs_phase,
            lbfgs_max_iter   = 500,
            rar_enabled      = rar_enabled,
            rar_interval     = rar_interval,
            rar_fraction     = rar_fraction,
            dtype            = dtype,
            grad_clip_norm   = grad_clip,
            use_jit          = use_jit,
        )

        cfg = RuntimeConfig(
            problem  = spec,
            model    = m_cfg,
            training = t_cfg,
            meta     = meta,
        )

        if self.verbose:
            self._print_summary(cfg, kappa)

        return cfg

    # ------------------------------------------------------------------
    # Adaptive refinement helpers (callable from Trainer)
    # ------------------------------------------------------------------

    def rar_resample(
        self,
        config: RuntimeConfig,
        model_fn: Callable,
        current_points: "Tensor",
        backend: AbstractBackend,
    ) -> "Tensor":
        """
        Residual-Adaptive Refinement (RAR):
        Evaluates the PDE residual over a candidate pool, ranks by magnitude,
        and replaces the bottom fraction of collocation points with the
        highest-residual candidates.

        Parameters
        ----------
        config         : current RuntimeConfig
        model_fn       : callable u(x) → u
        current_points : existing collocation point set
        backend        : active backend

        Returns
        -------
        Updated collocation points tensor
        """
        from physai.core.pde_residual import build_residual

        domain   = config.problem.domain
        n        = config.training.n_collocation
        frac     = config.training.rar_fraction
        n_new    = int(n * frac)
        n_pool   = n_new * 10
        n_keep   = max(n - n_new, 1)

        # Sample candidate pool uniformly, respecting the configured precision
        candidates = _uniform_sample(backend, domain, n_pool, dtype=config.training.dtype)

        res_fn = build_residual(
            config.problem.pde_name,
            backend,
            **config.problem.extra_params,
        )

        def _residual_mag(pts: "Tensor") -> "Tensor":
            r = res_fn(model_fn, pts)
            return backend.mean(backend.square(r), axis=-1) \
                if len(r.shape) > 1 else backend.square(r)

        with _no_grad_context(backend):
            cand_mag = _residual_mag(candidates)
            # Score the existing collocation points too, so the lowest-
            # residual points are genuinely dropped, as documented.
            curr_mag = _residual_mag(current_points)

        cand_np  = backend.to_numpy(candidates)
        curr_np  = backend.to_numpy(current_points)
        cand_mag_np = backend.to_numpy(cand_mag)
        curr_mag_np = backend.to_numpy(curr_mag)

        top_cand_idx = cand_mag_np.argsort()[::-1][:n_new]
        top_curr_idx = curr_mag_np.argsort()[::-1][:n_keep]

        new_pts  = backend.tensor(cand_np[top_cand_idx])
        kept_pts = backend.tensor(curr_np[top_curr_idx])
        return backend.concatenate([kept_pts, new_pts], axis=0)

    # ------------------------------------------------------------------
    # Learning-rate schedule evaluation
    # ------------------------------------------------------------------

    def lr_at_step(self, config: RuntimeConfig, step: int) -> float:
        """Compute the scheduled learning rate at ``step``."""
        s    = config.training.scheduler
        base = config.training.learning_rate

        if s.name == "none":
            return base

        # Linear warm-up
        if step < s.warmup_steps:
            return base * (step + 1) / (s.warmup_steps + 1)

        step_after = step - s.warmup_steps
        total      = s.decay_steps - s.warmup_steps

        if s.name == "cosine":
            t   = min(step_after / max(total, 1), 1.0)
            lr  = s.min_lr + 0.5 * (base - s.min_lr) * (1.0 + math.cos(math.pi * t))
            return float(lr)

        if s.name == "step":
            n_decays = step_after // s.step_size
            return max(base * (s.gamma ** n_decays), s.min_lr)

        if s.name == "exponential":
            return max(base * math.exp(-s.gamma * step_after), s.min_lr)

        return base

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _print_summary(self, cfg: RuntimeConfig, kappa: float) -> None:
        t = cfg.training
        m = cfg.model
        print(
            f"\n{'='*60}\n"
            f"  PhysAI AutoOptimizer — {cfg.meta.name.upper()}\n"
            f"{'='*60}\n"
            f"  Stiffness proxy κ        : {kappa:.3f}\n"
            f"  Architecture             : {m.arch.upper()}\n"
            f"  Layer sizes              : {m.layer_sizes}\n"
            f"  Activation               : {m.activation}\n"
            f"  Residual connections     : {m.use_residual}\n"
            f"  Optimizer                : {t.optimizer_name}  lr={t.learning_rate:.2e}\n"
            f"  Scheduler                : {t.scheduler.name}\n"
            f"  Collocation pts          : {t.n_collocation}\n"
            f"  Boundary pts             : {t.n_bc_points}\n"
            f"  Loss weights             : {t.loss_weights}\n"
            f"  RAR enabled              : {t.rar_enabled}  (interval={t.rar_interval})\n"
            f"  L-BFGS phase             : {t.use_lbfgs_phase}\n"
            f"  Precision                : {t.dtype}\n"
            f"  JIT                      : {t.use_jit}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _uniform_sample(
    backend: AbstractBackend,
    domain: "DomainSpec",
    n: int,
    dtype: str = "float32",
) -> "Tensor":
    """Sample n points uniformly from the spatial-temporal domain.

    ``dtype`` should match ``TrainingConfig.dtype``; previously this always
    cast to float32 regardless of the caller's precision, silently
    discarding the float64 recommendation AutoOptimizer makes for stiff
    problems and reintroducing precision loss exactly where it was meant
    to be avoided.

    If ``domain.geometry`` is set, spatial points are drawn from the true
    shape via rejection sampling (``Geometry.sample_interior``) instead of
    the bounding box, so arbitrary (CSG / custom / mesh) domains are
    respected everywhere this helper is used — collocation sampling and
    RAR candidate pools alike.
    """
    import numpy as np
    np_dtype = np.float64 if dtype == "float64" else np.float32

    if domain.geometry is not None:
        spatial = domain.geometry.sample_interior(n).astype(np_dtype)
        cols = [spatial]
    else:
        cols = []
        for lo, hi in domain.bounds:
            cols.append(np.random.uniform(lo, hi, (n, 1)).astype(np_dtype))
        cols = [np.concatenate(cols, axis=1)] if len(cols) > 1 else cols

    if domain.time_domain is not None:
        t0, T = domain.time_domain
        n_have = cols[0].shape[0]
        t_col = np.random.uniform(t0, T, (n_have, 1)).astype(np_dtype)
        pts = np.concatenate([cols[0], t_col], axis=1)
    else:
        pts = cols[0]
    return backend.tensor(pts)


def _no_grad_context(backend: AbstractBackend):
    """Context manager that disables gradient tracking where supported."""
    if backend.name == "torch":
        import torch
        return torch.no_grad()

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *_): pass

    return _NullCtx()


__all__ = [
    "PDEMeta",
    "DomainSpec",
    "ProblemSpec",
    "RuntimeConfig",
    "ModelConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "AutoOptimizer",
]