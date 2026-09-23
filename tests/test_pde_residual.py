"""
tests/test_pde_residual.py

Comprehensive tests for every PDE registered in physai.core.pde_residual.
PDE_REGISTRY.

Two tiers, by design (see the module docstring notes inline for why):

Tier 1 — "zero-field" test (all 51 registry entries except eikonal):
    Every homogeneous PDE in this file is trivially satisfied by the
    identically-zero field (u ≡ 0 everywhere), given each residual's
    *default* source/body-force/reaction parameters. This is cheap and
    catches real structural bugs (wrong attribute names, wrong axis in a
    stack/concatenate, shape mismatches, missing backend methods — this
    suite is what caught `atan` being absent from every backend, breaking
    `dendritic_solidification`). It is a fast, mechanical check.

    IMPORTANT CAVEAT, made explicit rather than glossed over: because the
    zero field is *disconnected* from the input tensor in most natural
    constructions, PyTorch's autograd (with `allow_unused=True`, as this
    codebase's `TorchBackend.grad` uses) short-circuits every derivative
    to an exact zero *without walking through the residual's actual
    derivative-composition logic*. In other words: Tier 1 proves the code
    *runs* and returns a sanely-shaped, finite, all-zero tensor — it does
    NOT prove the Laplacian/Hessian/mixed-derivative wiring inside each
    residual is mathematically correct. A residual with e.g. a swapped
    axis in its Hessian-diagonal loop would still pass Tier 1.

Tier 2 — "nonzero smooth field" stress test (all 51 registry entries,
    including eikonal): a genuinely nontrivial, smooth (infinitely
    differentiable, bounded) per-channel field built from sums of sines,
    so gradients actually flow and every nested `backend.grad` call in
    the residual performs a *real* differentiation. This does not check
    the numeric value (no independent ground truth), but does check the
    result is finite and correctly shaped — this is what would catch a
    genuine axis/index bug that Tier 1's short-circuited zeros would miss.

Tier 3 — exact manufactured-solution tests for a curated set of classical,
    independently-verifiable PDEs (Poisson, Heat, Wave, Advection,
    Helmholtz, Klein-Gordon, Eikonal, KdV). These use closed-form
    solutions with hand-derived source terms / parameter choices so the
    analytic residual is exactly (or, for KdV, very nearly, given
    float32 third-derivative autodiff noise) zero — real correctness
    evidence for the equations most likely to be depended on directly.

Run with: pytest tests/test_pde_residual.py -v
"""
import numpy as np
import pytest
import torch

from physai.backends.torch_backend import TorchBackend
from physai.core.pde_residual import PDE_REGISTRY, build_residual

backend = TorchBackend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _points(n: int, cols: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-1.0, 1.0, size=(n, cols)).astype(np.float32)
    return backend.tensor(arr)


def _zero_model(n_out: int):
    """A model whose output is identically zero (see Tier 1 caveat above:
    this is disconnected from the input, so autograd short-circuits every
    derivative to exact zero via `allow_unused=True` rather than actually
    differentiating anything)."""
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return backend.zeros(tuple(x.shape[:-1]) + (n_out,))
    return _fn


def _smooth_nonzero_model(n_out: int):
    """A genuinely nontrivial, smooth, bounded per-channel field so every
    `backend.grad` call inside a residual does real work. Not tied to any
    known closed-form residual — used only to stress-test that derivative
    chains run without shape/NaN errors on nonzero, nonconstant input."""
    def _fn(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        base = sum(backend.sin(x[..., i] + 0.1 * i) for i in range(d))
        channels = [base * (1.0 + 0.1 * c) + 0.3 for c in range(n_out)]
        return backend.stack(channels, axis=-1) if n_out > 1 else channels[0][..., None]
    return _fn


def _assert_finite_and_shaped(r: torch.Tensor, n_points: int) -> None:
    assert torch.isfinite(r).all(), "residual contains NaN/Inf"
    assert r.shape[0] == n_points, f"residual batch dim {r.shape[0]} != {n_points}"


# ---------------------------------------------------------------------------
# Per-PDE configuration: registry_key -> (n_input_cols, n_model_outputs, kwargs)
#
# n_input_cols encodes both spatial dimensionality *and* whether a trailing
# time column is present — chosen per-PDE to respect any hardcoded column
# indices in the implementation (several residuals assume an exact number
# of columns, e.g. Burgers/KdV assume exactly [x, t]; NavierStokes/
# ShallowWater2D assume exactly [x, y, t]). See the class docstrings in
# pde_residual.py for the documented column layout each expects.
#
# `kwargs` overrides any default constructor parameter that would make the
# zero field NOT satisfy the equation (only two PDEs need this: GiererMeinhardt
# has a nonzero production term `rho_a` at u=0 by default; GrayScott has a
# nonzero feed term `F` at u=v=0 by default), or that would otherwise put the
# generic `_smooth_nonzero_model` field outside a PDE's valid physical domain
# (relativistic_fluid: with the default speed_of_light=1.0, the generic
# smooth-nonzero velocity channel routinely exceeds |u|=1, i.e. superluminal
# and undefined for the Lorentz factor in *any* correct formula — bumping
# speed_of_light keeps the same field's velocity comfortably subluminal
# without weakening what's being tested).
# ---------------------------------------------------------------------------

CONFIGS = {
    "poisson":                        (2, 1, {}),
    "laplace":                        (2, 1, {}),
    "heat":                           (3, 1, {}),
    "diffusion":                      (3, 1, {}),
    "wave":                           (3, 1, {}),
    "burgers":                        (2, 1, {}),
    "navier_stokes":                  (3, 3, {}),
    "advection":                      (3, 1, {"velocity": [1.0, 1.0]}),
    "helmholtz":                      (2, 1, {}),
    "allen_cahn":                     (2, 1, {}),
    "cahn_hilliard":                  (2, 2, {}),
    "schrodinger":                    (2, 2, {}),
    "klein_gordon":                   (3, 1, {}),
    "reaction_diffusion":             (3, 1, {}),
    "darcy":                          (2, 1, {}),
    "euler":                          (2, 3, {}),
    "biharmonic":                     (2, 1, {}),
    "stokes":                         (2, 3, {}),
    "nls":                            (2, 2, {}),
    "kdv":                            (2, 1, {}),
    "fokker_planck":                  (3, 1, {}),
    "nlse_2d":                        (3, 2, {}),
    "sine_gordon_2d":                 (3, 1, {}),
    "fisher_kpp":                     (3, 1, {}),
    "klein_gordon_nonlinear_2d":      (3, 1, {}),
    "fitzhugh_nagumo":                (3, 2, {}),
    "cahn_hilliard_2d":               (3, 2, {}),
    "kuramoto_sivashinsky":           (2, 1, {}),
    "kadomtsev_petviashvili":         (3, 1, {}),
    "porous_medium":                  (3, 1, {}),
    "biharmonic_steady":              (2, 1, {}),
    "boussinesq_wave":                (2, 1, {}),
    "euler_tricomi":                  (2, 1, {}),
    "fokker_planck_2d":               (3, 1, {}),
    "swift_hohenberg":                (3, 1, {}),
    "black_scholes_2d":               (3, 1, {}),
    "regularised_long_wave":          (2, 1, {}),
    "gierer_meinhardt":               (3, 2, {"rho_a": 0.0}),
    "gray_scott":                     (3, 2, {"F": 0.0, "k": 0.0}),
    "viscous_wave":                   (3, 1, {}),
    "radhakrishnan_kundu_lakshmanan": (2, 2, {}),
    "complex_ginzburg_landau_2d":     (3, 2, {}),
    "drift_diffusion_poisson":        (2, 2, {}),
    "shallow_water_2d":               (3, 3, {}),
    "heston_volatility":              (3, 1, {}),
    "phase_field_crystal":            (3, 1, {}),
    "brinkman_darcy":                 (2, 3, {}),
    "perona_malik":                   (3, 1, {}),
    "burgers_2d":                     (3, 2, {}),
    "boussinesq_convection":          (3, 4, {}),
    "dendritic_solidification":       (3, 2, {}),
    "relativistic_fluid":             (2, 2, {"speed_of_light": 10.0}),

    # --- Sprint additions (see core/pde_residual.py) ---
    "phonon":                         (2, 1, {}),
    "phonons":                        (2, 1, {}),
    "dirac":                          (2, 4, {}),
    "dirac_equation":                 (2, 4, {}),
    "bose_einstein":                  (2, 2, {}),
    "gross_pitaevskii":               (2, 2, {}),
}

# Excluded from the generic zero-field/smooth-field tiers (like eikonal)
# because they have a genuine domain singularity at u=0 or at a generic
# sign-changing field, not a residual bug: EFE needs an invertible,
# correctly-signed metric (zero/generic g is degenerate); fermi_gas's
# n^(5/3) closure is singular at n=0 and undefined for n<0 (which the
# generic sin-based smooth field routinely produces); quantum_relativistic
# _fluid's Bohm term divides by sqrt(h), singular at h=0 and undefined for
# h<0 for the same reason. Each gets its own dedicated positive-field test
# in Tier 3b below instead.
_DOMAIN_RESTRICTED_KEYS = {
    "einstein_field", "efe", "fermi_gas", "fermi_dirac", "quantum_relativistic_fluid",
}

# eikonal is excluded from CONFIGS/Tier 1: its default speed_fn=1 means the
# zero field does NOT satisfy |grad u| = 1 (grad of the zero field is zero,
# not magnitude 1), so it gets its own manufactured-solution test in Tier 3
# and a dedicated Tier-2-style stress entry below.
EIKONAL_CONFIG = (2, 1, {})

ALL_REGISTRY_KEYS = sorted((set(PDE_REGISTRY) | {"eikonal"}) - _DOMAIN_RESTRICTED_KEYS)


def test_every_registry_key_has_a_test_config():
    """Guards against silently skipping coverage if a new PDE is added to
    PDE_REGISTRY without a matching entry here."""
    configured = set(CONFIGS) | {"eikonal"} | _DOMAIN_RESTRICTED_KEYS
    missing = set(PDE_REGISTRY) - configured
    assert not missing, (
        f"PDE_REGISTRY has keys with no test configuration: {sorted(missing)}. "
        f"Add an entry to CONFIGS in test_pde_residual.py."
    )


# ---------------------------------------------------------------------------
# Tier 1 — zero-field trivial-solution test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(CONFIGS))
def test_zero_field_satisfies_homogeneous_pde(key):
    cols, n_out, kwargs = CONFIGS[key]
    residual = build_residual(key, backend, **kwargs)
    pts = _points(6, cols)
    model = _zero_model(n_out)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5), (
        f"'{key}': zero field should exactly satisfy this homogeneous PDE "
        f"with default parameters, got max|residual|={r.abs().max().item()}"
    )

    loss = residual.loss(model, pts)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


# ---------------------------------------------------------------------------
# Tier 2 — nonzero smooth-field stress test (every registered PDE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ALL_REGISTRY_KEYS)
def test_nonzero_smooth_field_runs_and_is_finite(key):
    if key == "eikonal":
        cols, n_out, kwargs = EIKONAL_CONFIG
    else:
        cols, n_out, kwargs = CONFIGS[key]

    residual = build_residual(key, backend, **kwargs)
    pts = _points(6, cols, seed=1)
    model = _smooth_nonzero_model(n_out)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)

    loss = residual.loss(model, pts)
    assert torch.isfinite(loss).all()
    assert loss.numel() == 1


# ---------------------------------------------------------------------------
# Tier 3 — exact manufactured-solution tests
# ---------------------------------------------------------------------------

def test_poisson_quadratic_manufactured_solution():
    """u = x0^2 + x1^2  =>  Δu = 4  =>  need f = -4 for  -Δu - f = 0."""
    residual = build_residual(
        "poisson", backend,
        source_fn=lambda x: -4.0 * backend.ones(x.shape[:-1]),
    )
    pts = _points(8, 2, seed=2)
    model = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2)[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_heat_linear_in_time_manufactured_solution():
    """u = x0 + t  =>  u_t = 1, Δu = 0  =>  need f = 1 for  u_t - αΔu - f = 0."""
    residual = build_residual(
        "heat", backend, alpha=1.0,
        source_fn=lambda xt: backend.ones(xt.shape[:-1]),
    )
    pts = _points(8, 3, seed=3)
    model = lambda x: (x[..., 0] + x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_wave_quadratic_in_time_manufactured_solution():
    """u = t^2  =>  u_tt = 2, Δu = 0  =>  need f = 2 for  u_tt - c²Δu - f = 0.

    Exercises the *second* time derivative specifically (nested grad
    indexing), which the zero-field test cannot validate.
    """
    residual = build_residual(
        "wave", backend, c=1.0,
        source_fn=lambda xt: 2.0 * backend.ones(xt.shape[:-1]),
    )
    pts = _points(8, 3, seed=4)
    model = lambda x: (x[..., -1] ** 2)[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_advection_traveling_wave_manufactured_solution():
    """u = sum(x_i) - (sum(c_i)) t exactly solves u_t + c·∇u = 0 for any
    constant velocity c, since ∇u = (1,...,1) and u_t = -sum(c_i)."""
    velocity = [1.0, 2.0]
    residual = build_residual("advection", backend, velocity=velocity)
    pts = _points(8, 3, seed=5)
    total_c = sum(velocity)
    model = lambda x: (x[..., 0] + x[..., 1] - total_c * x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_helmholtz_plane_wave_manufactured_solution():
    """u = sin(x0), k = 1  =>  Δu = -sin(x0) = -u  =>  Δu + k²u = 0 exactly."""
    residual = build_residual("helmholtz", backend, k=1.0)
    pts = _points(8, 2, seed=6)
    model = lambda x: backend.sin(x[..., 0])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_klein_gordon_massless_traveling_wave_manufactured_solution():
    """u = cos(x0 - t), c = 1, m = 0  =>  u_tt = -u, Δu = -u  =>
    u_tt - c²Δu + m²u = -u - (-u) + 0 = 0."""
    residual = build_residual("klein_gordon", backend, c=1.0, m=0.0)
    pts = _points(8, 2, seed=7)
    model = lambda x: backend.cos(x[..., 0] - x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3)


def test_eikonal_identity_solution():
    """u = x0  =>  |∇u| = 1, matching the default speed_fn ≡ 1 exactly
    (up to the implementation's own +1e-8 regularizer inside the sqrt)."""
    residual = build_residual("eikonal", backend)
    pts = _points(8, 2, seed=8)
    model = lambda x: x[..., 0:1]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-4)


def test_kdv_soliton_manufactured_solution():
    """
    Classic KdV 1-soliton exact solution for ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0:

        u(x, t) = (c/2) sech²( (√c / 2)(x - c t) )

    Expressed via sech²(z) = 1 - tanh²(z) since the backend interface has
    no cosh/sech but does have tanh. This exercises the *real* third
    spatial derivative chain (unlike the zero-field test).
    """
    residual = build_residual("kdv", backend)
    pts = _points(10, 2, seed=9)
    c = 4.0
    sqrt_c = c ** 0.5

    def model(x):
        z = (sqrt_c / 2.0) * (x[..., 0] - c * x[..., -1])
        sech2 = 1.0 - backend.square(torch.tanh(z))
        return ((c / 2.0) * sech2)[..., None]

    r = residual(model, pts)
    # float32 third-derivative autodiff through tanh has some numerical
    # noise; this tolerance is loose relative to Tier-3's other tests but
    # still tight enough to catch a genuinely wrong derivative chain.
    assert torch.allclose(r, torch.zeros_like(r), atol=2e-2), (
        f"KdV soliton residual too large: max|r|={r.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Tier 3 (extended) — manufactured/exact solutions for the remaining
# registry entries, so Tier 3 covers all 51 registered PDEs (+ eikonal)
# instead of the original curated 8. Every closed form below was derived
# and symbolically verified with sympy before being transcribed here (see
# development notes); each docstring states the exact solution used.
# ---------------------------------------------------------------------------

_ATOL = 1e-3


def test_biharmonic_quartic_manufactured_solution():
    """u = x0^4 + x1^4  =>  Delta^2 u = 48  =>  f = 48."""
    residual = build_residual(
        "biharmonic", backend,
        source_fn=lambda x: 48.0 * backend.ones(x.shape[:-1]),
    )
    pts = _points(8, 2, seed=20)
    model = lambda x: (x[..., 0] ** 4 + x[..., 1] ** 4)[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_biharmonic_steady_quartic_manufactured_solution():
    """Same manufactured solution as biharmonic, via the steady 2-D variant."""
    residual = build_residual(
        "biharmonic_steady", backend,
        source_fn=lambda x: 48.0 * backend.ones(x.shape[:-1]),
    )
    pts = _points(8, 2, seed=21)
    model = lambda x: (x[..., 0] ** 4 + x[..., 1] ** 4)[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_darcy_quadratic_manufactured_solution():
    """K(x)=1 (default) reduces Darcy to Poisson: u=x0^2+x1^2 => f=-4."""
    residual = build_residual(
        "darcy", backend,
        source_fn=lambda x: -4.0 * backend.ones(x.shape[:-1]),
    )
    pts = _points(8, 2, seed=22)
    model = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2)[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_euler_tricomi_bilinear_manufactured_solution():
    """u = a*x + b*y + c*x*y has u_xx=u_yy=0, exactly solving
    u_yy - x*u_xx = 0 for any a,b,c."""
    residual = build_residual("euler_tricomi", backend)
    pts = _points(8, 2, seed=23)
    model = lambda x: (2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.5 * x[..., 0] * x[..., 1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_burgers_rational_manufactured_solution():
    """u = x/(1+t) has u_xx=0, and u_t + u*u_x = 0 exactly (independent of
    nu), so it exactly solves viscous Burgers for any nu."""
    residual = build_residual("burgers", backend, nu=0.05)
    pts = _points(8, 2, seed=24)
    model = lambda x: (x[..., 0] / (1.0 + x[..., -1]))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_fokker_planck_polynomial_manufactured_solution():
    """Default drift=0, D=1: p = x0^2 + x1^2 + 2t exactly solves
    p_t - 0.5*Delta(p) = 0 (Delta p = 4, p_t = 2)."""
    residual = build_residual("fokker_planck", backend)
    pts = _points(8, 3, seed=25)
    model = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2 + 2.0 * x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_schrodinger_free_particle_plane_wave_manufactured_solution():
    """Free particle (V=0 default): psi = exp(i(kx - wt)), w = hbar*k^2/(2m)
    exactly solves the linear Schrodinger equation."""
    hbar, mass, k = 1.0, 1.0, 1.5
    w = hbar * k ** 2 / (2.0 * mass)
    residual = build_residual("schrodinger", backend, hbar=hbar, mass=mass)
    pts = _points(8, 2, seed=26)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([torch.cos(phase), torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_allen_cahn_tanh_front_manufactured_solution():
    """Steady heteroclinic front u = tanh(x0 / (sqrt(2)*eps)) exactly solves
    eps^2*Delta(u) + u(1-u^2) = 0, i.e. u_t=0 for this steady solution."""
    eps = 0.2
    residual = build_residual("allen_cahn", backend, epsilon=eps)
    pts = _points(8, 2, seed=27)
    model = lambda x: torch.tanh(x[..., 0] / (2.0 ** 0.5 * eps))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_cahn_hilliard_tanh_front_manufactured_solution():
    """The same Allen-Cahn tanh front is also an exact (steady, w=0) solution
    of Cahn-Hilliard: w = -eps^2*Delta(u) + u^3 - u = 0 identically."""
    eps = 0.2
    residual = build_residual("cahn_hilliard", backend, epsilon=eps)
    pts = _points(8, 2, seed=28)

    def model(x):
        u = torch.tanh(x[..., 0] / (2.0 ** 0.5 * eps))
        w = torch.zeros_like(u)
        return backend.stack([u, w], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_cahn_hilliard_2d_tanh_front_manufactured_solution():
    """Same tanh-front trick, with gamma playing the role of eps^2."""
    gamma = 0.04
    residual = build_residual("cahn_hilliard_2d", backend, gamma=gamma)
    pts = _points(8, 3, seed=29)

    def model(x):
        c = torch.tanh(x[..., 0] / (2.0 ** 0.5 * gamma ** 0.5))
        mu = torch.zeros_like(c)
        return backend.stack([c, mu], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_kuramoto_sivashinsky_rational_manufactured_solution():
    """u = x/(1+t): u_xx=u_xxxx=0 and u_t+u*u_x=0 exactly (same identity as
    the Burgers manufactured solution), so KS is solved for any alpha,beta."""
    residual = build_residual("kuramoto_sivashinsky", backend, alpha=1.0, beta=1.0)
    pts = _points(8, 2, seed=30)
    model = lambda x: (x[..., 0] / (1.0 + x[..., -1]))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_kadomtsev_petviashvili_soliton_manufactured_solution():
    """A y-independent KdV 1-soliton exactly zeroes the KdV core term
    everywhere, so d/dx(core)=0 and u_yy=0 (y-independent) => KP residual=0
    for any lambda."""
    residual = build_residual("kadomtsev_petviashvili", backend, lam=1.0)
    pts = _points(8, 3, seed=31)
    c = 4.0
    sqrt_c = c ** 0.5

    def model(x):
        z = (sqrt_c / 2.0) * (x[..., 0] - c * x[..., -1])
        sech2 = 1.0 - backend.square(torch.tanh(z))
        return ((c / 2.0) * sech2)[..., None]

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=3e-2), (
        f"KP soliton residual too large: max|r|={r.abs().max().item()}"
    )


def test_porous_medium_steady_power_manufactured_solution():
    """u^m = x0 + 2 (linear, hence harmonic) and steady (u_t=0) exactly
    solves u_t - Delta(u^m) = 0."""
    m = 2.0
    residual = build_residual("porous_medium", backend, m=m)
    pts = _points(8, 3, seed=32)
    model = lambda x: ((x[..., 0] + 2.0) ** (1.0 / m))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_perona_malik_linear_ramp_manufactured_solution():
    """A linear (affine) field has constant gradient, hence constant flux
    c(|grad u|^2)*grad(u), whose divergence is zero — an exact steady
    solution for any K."""
    residual = build_residual("perona_malik", backend, K=0.3)
    pts = _points(8, 3, seed=33)
    model = lambda x: (0.4 * x[..., 0] - 0.7 * x[..., 1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_black_scholes_2d_linear_payoff_manufactured_solution():
    """V = a*S1 + b*S2 (a linear payoff / delta-one position) exactly solves
    the Black-Scholes PDE for any r, sigma1, sigma2, rho — every
    second-order and cross term vanishes and the first-order/-rV terms
    cancel identically."""
    residual = build_residual("black_scholes_2d", backend, r=0.05, sigma1=0.2, sigma2=0.3, rho=0.4)
    pts = _points(8, 3, seed=34)
    model = lambda x: (1.3 * x[..., 0] - 0.6 * x[..., 1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_heston_volatility_linear_in_price_manufactured_solution():
    """U = a*S exactly solves the Heston PDE (the classic "delta-one"
    solution: r*S*U_S cancels r*U identically, and every variance-dependent
    term vanishes since U doesn't depend on v)."""
    residual = build_residual("heston_volatility", backend, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    pts = _points(8, 3, seed=35)
    model = lambda x: (2.5 * x[..., 0])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_regularised_long_wave_manufactured_solution():
    """u = (x - t) / (1 + t) exactly solves u_t + u_x + u*u_x - mu*u_xxt = 0
    for any mu (u_xx = 0 identically, so the mixed term vanishes too)."""
    residual = build_residual("regularised_long_wave", backend, mu=0.2)
    pts = _points(8, 2, seed=36)
    model = lambda x: ((x[..., 0] - x[..., -1]) / (1.0 + x[..., -1]))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_viscous_wave_damped_standing_wave_manufactured_solution():
    """u = exp(-alpha t)*cos(k x) solves u_tt - c^2*Delta(u) - nu*Delta(u_t)=0
    exactly when alpha^2 - alpha*k^2*nu + c^2*k^2 = 0. nu is chosen large
    enough (relative to c, k) that alpha is real."""
    c, nu, k = 1.0, 3.0, 1.0
    disc = (k ** 2 * nu) ** 2 - 4.0 * (c * k) ** 2
    alpha = (k ** 2 * nu + disc ** 0.5) / 2.0
    residual = build_residual("viscous_wave", backend, c=c, nu=nu)
    pts = _points(8, 3, seed=37)
    model = lambda x: (torch.exp(-alpha * x[..., -1]) * torch.cos(k * x[..., 0]))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_radhakrishnan_kundu_lakshmanan_plane_wave_manufactured_solution():
    """u=A cos(kx-wt), v=A sin(kx-wt) exactly solves RKL when
    w = alpha*k^2 - gamma*k^3 - beta*A^2 (both real/imag equations share
    the same dispersion relation)."""
    alpha, beta, gamma, A, k = 0.5, 1.0, 0.1, 0.7, 1.2
    w = alpha * k ** 2 - gamma * k ** 3 - beta * A ** 2
    residual = build_residual("radhakrishnan_kundu_lakshmanan", backend, alpha=alpha, beta=beta, gamma=gamma)
    pts = _points(8, 2, seed=38)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([A * torch.cos(phase), A * torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_complex_ginzburg_landau_2d_plane_wave_manufactured_solution():
    """The classic CGLE plane-wave solution u=A cos(kx-wt), v=A sin(kx-wt)
    (y-independent) is exact when A^2 = 1 - k^2 and w = c1*k^2 + A^2*c3."""
    c1, c3, k = 0.5, 1.0, 0.5
    A = (1.0 - k ** 2) ** 0.5
    w = c1 * k ** 2 + A ** 2 * c3
    residual = build_residual("complex_ginzburg_landau_2d", backend, c1=c1, c3=c3)
    pts = _points(8, 3, seed=39)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([A * torch.cos(phase), A * torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_nlse_2d_plane_wave_manufactured_solution():
    """u=A cos(kx-wt), v=A sin(kx-wt) (y-independent) exactly solves the 2D
    NLSE when w = k^2 - beta*A^2."""
    beta, A, k = 1.0, 0.8, 1.1
    w = k ** 2 - beta * A ** 2
    residual = build_residual("nlse_2d", backend, beta=beta)
    pts = _points(8, 3, seed=40)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([A * torch.cos(phase), A * torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_nls_plane_wave_manufactured_solution():
    """1-D NLS plane wave psi=A*exp(i(kx-wt)) is exact when
    w = hbar*k^2/(2m) + g*A^2/hbar."""
    hbar, mass, g, A, k = 1.0, 1.0, 1.0, 0.6, 1.3
    w = hbar * k ** 2 / (2.0 * mass) + g * A ** 2 / hbar
    residual = build_residual("nls", backend, hbar=hbar, mass=mass, g=g)
    pts = _points(8, 2, seed=41)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([A * torch.cos(phase), A * torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_drift_diffusion_poisson_manufactured_solution():
    """With mu_n=0 (electron drift term switched off), n=n0 (constant)
    trivially solves the electron-continuity equation, while
    psi = -(scale*n0/2)*x^2 is a genuinely nonconstant exact solution of
    the coupled Poisson equation -Delta(psi) = scale*n0."""
    n0, scale = 1.5, 2.0
    residual = build_residual("drift_diffusion_poisson", backend, D_n=1.0, mu_n=0.0, scaling_factor=scale)
    pts = _points(8, 2, seed=42)

    def model(x):
        n = n0 * torch.ones_like(x[..., 0])
        psi = -(scale * n0 / 2.0) * x[..., 0] ** 2
        return backend.stack([n, psi], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_brinkman_darcy_harmonic_conjugate_manufactured_solution():
    """u=exp(kx)cos(ky), v=-exp(kx)sin(ky) is a harmonic-conjugate
    (potential-flow) pair: both are harmonic (Delta=0) and satisfy
    continuity exactly; p=-(darcy/k)*exp(kx)cos(ky) makes the momentum
    balance exact too, for any mu_eff."""
    mu_eff, darcy, k = 0.05, 10.0, 1.0
    residual = build_residual("brinkman_darcy", backend, mu_eff=mu_eff, darcy_coeff=darcy)
    pts = _points(8, 2, seed=43)

    def model(x):
        ekx = torch.exp(k * x[..., 0])
        u = ekx * torch.cos(k * x[..., 1])
        v = -ekx * torch.sin(k * x[..., 1])
        p = -(darcy / k) * ekx * torch.cos(k * x[..., 1])
        return backend.stack([u, v, p], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_burgers_2d_rational_manufactured_solution():
    """u=x/(1+t), v=y/(1+t): each satisfies the 1-D Burgers identity
    independently (u_y=v_x=0 decouples the cross-advection terms), so both
    component equations vanish exactly for any nu."""
    residual = build_residual("burgers_2d", backend, nu=0.02)
    pts = _points(8, 3, seed=44)

    def model(x):
        t = x[..., -1]
        u = x[..., 0] / (1.0 + t)
        v = x[..., 1] / (1.0 + t)
        return backend.stack([u, v], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_boussinesq_convection_strain_flow_manufactured_solution():
    """u=x, v=-y (potential strain flow, satisfies continuity and the pure
    NS part with p=-(1/2)(x^2+y^2)); a constant temperature field T=T0
    trivially solves the advection-diffusion equation, and adding
    +Ra*Pr*T0*y to the pressure exactly cancels the buoyancy term."""
    Pr, Ra, T0 = 0.71, 1000.0, 0.3
    residual = build_residual("boussinesq_convection", backend, Pr=Pr, Ra=Ra)
    pts = _points(8, 3, seed=45)

    def model(x):
        xx, yy = x[..., 0], x[..., 1]
        u, v = xx, -yy
        p = -0.5 * (xx ** 2 + yy ** 2) + Ra * Pr * T0 * yy
        T = T0 * torch.ones_like(xx)
        return backend.stack([u, v, p, T], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_boussinesq_wave_soliton_manufactured_solution():
    """The classic 'good Boussinesq' soliton u = 2k^2 * sech^2(k(x-ct))
    exactly solves u_tt - u_xx - 3(u^2)_xx - u_xxxx = 0 when c^2 = 4k^2+1."""
    k = 0.5
    A = 2.0 * k ** 2
    c = (4.0 * k ** 2 + 1.0) ** 0.5
    residual = build_residual("boussinesq_wave", backend)
    pts = _points(8, 2, seed=46)

    def model(x):
        z = k * (x[..., 0] - c * x[..., -1])
        sech2 = 1.0 - backend.square(torch.tanh(z))
        return (A * sech2)[..., None]

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=5e-2), (
        f"Boussinesq soliton residual too large: max|r|={r.abs().max().item()}"
    )


def test_navier_stokes_potential_strain_flow_manufactured_solution():
    """u=x, v=-y is a divergence-free potential-flow strain field; with
    p = -(rho/2)(x^2+y^2), it exactly solves steady incompressible NS
    (u_t=v_t=0) for any rho, mu."""
    rho, mu = 1.2, 0.05
    residual = build_residual("navier_stokes", backend, rho=rho, mu=mu)
    pts = _points(8, 3, seed=47)

    def model(x):
        xx, yy = x[..., 0], x[..., 1]
        u, v = xx, -yy
        p = -0.5 * rho * (xx ** 2 + yy ** 2)
        return backend.stack([u, v, p], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_stokes_potential_strain_flow_manufactured_solution():
    """The same strain flow u=x, v=-y (both harmonic, div-free) solves
    steady Stokes flow with p=0 for any mu, since there's no convective
    term to balance."""
    residual = build_residual("stokes", backend, mu=1.0)
    pts = _points(8, 2, seed=48)

    def model(x):
        xx, yy = x[..., 0], x[..., 1]
        u, v = xx, -yy
        p = torch.zeros_like(xx)
        return backend.stack([u, v, p], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_sine_gordon_2d_kink_manufactured_solution():
    """The classic 1-D sine-Gordon kink u=4*atan(exp((x-ct)/sqrt(1-c^2)))
    embedded y-independently exactly solves the 2-D equation (Delta
    reduces to u_xx since u_yy=0)."""
    c = 0.5
    residual = build_residual("sine_gordon_2d", backend)
    pts = _points(8, 3, seed=49)

    def model(x):
        xi = (x[..., 0] - c * x[..., -1]) / (1.0 - c ** 2) ** 0.5
        return (4.0 * torch.atan(torch.exp(xi)))[..., None]

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_klein_gordon_nonlinear_2d_massive_plane_wave_manufactured_solution():
    """With gamma=0 the cubic term drops out and the equation reduces to
    linear Klein-Gordon; u=cos(x - w t), w=sqrt(1+m^2) (y-independent)
    solves it exactly."""
    m = 0.7
    w = (1.0 + m ** 2) ** 0.5
    residual = build_residual("klein_gordon_nonlinear_2d", backend, m=m, gamma=0.0)
    pts = _points(8, 3, seed=50)
    model = lambda x: torch.cos(x[..., 0] - w * x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2)


def test_reaction_diffusion_heat_mode_manufactured_solution():
    """With reaction_fn overridden to zero, the equation reduces to plain
    diffusion; u = sin(x0)*exp(-D*t) exactly solves u_t - D*Delta(u) = 0."""
    D = 0.3
    residual = build_residual("reaction_diffusion", backend, D=D, reaction_fn=lambda u: 0.0 * u)
    pts = _points(8, 3, seed=51)
    model = lambda x: (torch.sin(x[..., 0]) * torch.exp(-D * x[..., -1]))[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


# ---------------------------------------------------------------------------
# Tier 3b — nonzero constant-equilibrium checks.
#
# For the remaining registry entries (fully coupled nonlinear systems with
# no tractable closed-form non-constant solution and no source/forcing
# hook to inject one via MMS), a spatially-and-temporally constant field
# reduces every derivative term to exactly zero, leaving a pure algebraic
# equation in the constant(s) — solved by hand below. This still exercises
# the nonlinear/algebraic term evaluation with a genuinely nonzero value
# (unlike Tier 1's disconnected zero field), but — exactly like Tier 1 —
# it does NOT exercise the derivative-composition logic, since a constant
# model output is disconnected from the input in the same way the zero
# field is (autograd short-circuits every derivative to zero without
# walking through the residual's actual derivative code). Kept as a
# separate, explicitly-labelled tier rather than mixed into Tier 3 so that
# distinction stays visible.
# ---------------------------------------------------------------------------

def test_fisher_kpp_saturated_equilibrium():
    """u=1 (spatially/temporally constant) is an equilibrium of the
    logistic reaction term r*u*(1-u) for any D, r."""
    residual = build_residual("fisher_kpp", backend, D=0.1, r=1.0)
    pts = _points(6, 3, seed=52)
    model = lambda x: torch.ones_like(x[..., 0:1])  # noqa: E731
    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_fitzhugh_nagumo_constant_equilibrium():
    """v=v0, w=(a/b)*v0 solves the w-equation exactly for any a,b; choosing
    I = a*v0/b + v0^3 - v0 makes the v-equation vanish too."""
    a, b, v0 = 0.5, 0.8, 0.3
    w0 = (a / b) * v0
    I = a * v0 / b + v0 ** 3 - v0
    residual = build_residual("fitzhugh_nagumo", backend, a=a, b=b, I=I)
    pts = _points(6, 3, seed=53)

    def model(x):
        v = v0 * torch.ones_like(x[..., 0])
        w = w0 * torch.ones_like(x[..., 0])
        return backend.stack([v, w], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_gierer_meinhardt_constant_equilibrium():
    """a=a0, i=a0^2/mu_i solves the inhibitor equation exactly; choosing
    rho_a = a0*mu_a - mu_i makes the activator equation vanish too."""
    a0, mu_a, mu_i = 0.5, 0.01, 0.02
    i0 = a0 ** 2 / mu_i
    rho_a = a0 * mu_a - mu_i
    residual = build_residual("gierer_meinhardt", backend, mu_a=mu_a, mu_i=mu_i, rho_a=rho_a)
    pts = _points(6, 3, seed=54)

    def model(x):
        a = a0 * torch.ones_like(x[..., 0])
        i = i0 * torch.ones_like(x[..., 0])
        return backend.stack([a, i], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_gray_scott_trivial_equilibrium():
    """(u, v) = (1, 0) is an equilibrium of the Gray-Scott reaction terms
    for any F, k."""
    residual = build_residual("gray_scott", backend, F=0.035, k=0.06)
    pts = _points(6, 3, seed=55)

    def model(x):
        u = torch.ones_like(x[..., 0])
        v = torch.zeros_like(x[..., 0])
        return backend.stack([u, v], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_swift_hohenberg_nonzero_equilibrium():
    """u0^2 = r-1 is a nonzero constant equilibrium of the Swift-Hohenberg
    reaction term for any r > 1."""
    r_param = 2.0
    u0 = (r_param - 1.0) ** 0.5
    residual = build_residual("swift_hohenberg", backend, r=r_param)
    pts = _points(6, 3, seed=56)
    model = lambda x: (u0 * torch.ones_like(x[..., 0]))[..., None]  # noqa: E731
    res = residual(model, pts)
    _assert_finite_and_shaped(res, 6)
    assert torch.allclose(res, torch.zeros_like(res), atol=_ATOL)


def test_phase_field_crystal_constant_equilibrium():
    """Any spatially/temporally constant phi solves the equation exactly:
    every Laplacian term vanishes, leaving phi_t - Delta(mu) = 0 - 0 = 0."""
    residual = build_residual("phase_field_crystal", backend, epsilon=0.25)
    pts = _points(6, 3, seed=57)
    model = lambda x: (0.4 * torch.ones_like(x[..., 0]))[..., None]  # noqa: E731
    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_dendritic_solidification_constant_equilibrium():
    """phi=1 zeroes the phi(1-phi)(...) reaction term regardless of T (and
    of the atan() nonlinearity's value), and phi_t=0 trivially satisfies
    the temperature equation too."""
    residual = build_residual("dendritic_solidification", backend, epsilon=0.01, K=1.0)
    pts = _points(6, 3, seed=58)

    def model(x):
        phi = torch.ones_like(x[..., 0])
        T = 0.3 * torch.ones_like(x[..., 0])
        return backend.stack([phi, T], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_relativistic_fluid_constant_state():
    """The relativistic fluid residual is purely a sum of time/space
    derivatives (no bare algebraic term), so any constant (h0, u0) solves
    it exactly."""
    residual = build_residual("relativistic_fluid", backend, speed_of_light=1.0)
    pts = _points(6, 2, seed=59)

    def model(x):
        h = 1.3 * torch.ones_like(x[..., 0])
        u = 0.2 * torch.ones_like(x[..., 0])
        return backend.stack([h, u], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_fokker_planck_2d_constant_equilibrium():
    """A spatially/temporally constant probability density p solves the
    equation exactly regardless of the (nonzero) drift field: p_t=0,
    grad(p)=0 so the divergence-of-drift term vanishes too, and the
    Laplacian of a constant is 0."""
    residual = build_residual("fokker_planck_2d", backend, drift=(0.1, 0.1), D=0.05)
    pts = _points(6, 3, seed=62)
    model = lambda x: (0.5 * torch.ones_like(x[..., 0]))[..., None]  # noqa: E731
    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_euler_compressible_uniform_state():
    """A spatially/temporally uniform gas state (rho0, u0, E0) trivially
    solves the compressible Euler conservation laws (every flux gradient
    and time derivative is zero)."""
    residual = build_residual("euler", backend, gamma=1.4)
    pts = _points(6, 2, seed=60)

    def model(x):
        rho = 1.0 * torch.ones_like(x[..., 0])
        u = 0.1 * torch.ones_like(x[..., 0])
        E = 2.5 * torch.ones_like(x[..., 0])
        return backend.stack([rho, u, E], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_shallow_water_2d_rest_state():
    """A lake-at-rest state (h0 constant, u=v=0) trivially solves the
    shallow-water equations (no bathymetry forcing term is present)."""
    residual = build_residual("shallow_water_2d", backend, g=9.81)
    pts = _points(6, 3, seed=61)

    def model(x):
        h = 2.0 * torch.ones_like(x[..., 0])
        u = torch.zeros_like(x[..., 0])
        v = torch.zeros_like(x[..., 0])
        return backend.stack([h, u, v], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_phonon_dispersive_plane_wave_manufactured_solution():
    """u = cos(kx - wt) exactly solves the dispersive lattice wave eq.
    when w^2 = c^2 k^2 + beta*a^2*k^4 (its exact dispersion relation)."""
    K_spring, m, a, k = 2.0, 1.0, 0.3, 1.2
    c_sq = K_spring * a ** 2 / m
    beta = c_sq / 12.0
    w = (c_sq * k ** 2 + beta * a ** 2 * k ** 4) ** 0.5
    residual = build_residual("phonon", backend, mass=m, spring_constant=K_spring, lattice_spacing=a)
    pts = _points(8, 2, seed=70)
    model = lambda x: torch.cos(k * x[..., 0] - w * x[..., -1])[..., None]  # noqa: E731
    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2), f"max|r|={r.abs().max().item()}"


def test_dirac_free_particle_plane_wave_manufactured_solution():
    """Standard free-particle Dirac plane wave: psi1=A e^{i(kx-Et)},
    psi2=B e^{i(kx-Et)} with E^2 = m^2c^4 + c^2k^2 and B=(E-mc^2)A/(ck)
    exactly solves the 1+1-D Dirac equation (hbar=1)."""
    m, c, k = 1.0, 1.0, 1.0
    E = (m ** 2 * c ** 4 + c ** 2 * k ** 2) ** 0.5
    A = 1.0
    B = (E - m * c ** 2) * A / (c * k)
    residual = build_residual("dirac", backend, mass=m, speed_of_light=c, hbar=1.0)
    pts = _points(8, 2, seed=71)

    def model(x):
        theta = k * x[..., 0] - E * x[..., -1]
        u1, v1 = A * torch.cos(theta), A * torch.sin(theta)
        u2, v2 = B * torch.cos(theta), B * torch.sin(theta)
        return backend.stack([u1, v1, u2, v2], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-2), f"max|r|={r.abs().max().item()}"


def test_bose_einstein_free_particle_plane_wave_manufactured_solution():
    """With the trap and interaction switched off (V=0, g=0), GPE reduces
    to the plain linear Schrodinger equation, so the same free-particle
    plane wave used for `schrodinger` above solves it exactly."""
    hbar, mass, k = 1.0, 1.0, 1.5
    w = hbar * k ** 2 / (2.0 * mass)
    residual = build_residual(
        "bose_einstein", backend, hbar=hbar, mass=mass, interaction_g=0.0,
        trap_fn=lambda x: torch.zeros_like(x[..., 0]),
    )
    pts = _points(8, 2, seed=72)

    def model(x):
        phase = k * x[..., 0] - w * x[..., -1]
        return backend.stack([torch.cos(phase), torch.sin(phase)], axis=-1)

    r = residual(model, pts)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_fermi_gas_uniform_equilibrium():
    """A spatially/temporally constant, strictly positive density n0 with
    u=0 trivially solves the continuity+momentum equations (every
    derivative term is zero; the n^(5/3) pressure term only appears
    inside a spatial derivative, which vanishes for constant n)."""
    residual = build_residual("fermi_gas", backend, mass=1.0, hbar=1.0)
    pts = _points(6, 2, seed=73)

    def model(x):
        n = 0.8 * torch.ones_like(x[..., 0])
        u = torch.zeros_like(x[..., 0])
        return backend.stack([n, u], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_quantum_relativistic_fluid_uniform_equilibrium():
    """A constant, strictly positive enthalpy h0 with u=0: every
    time/space derivative term vanishes (including the Bohm term, since
    Laplacian of a constant sqrt(h0) is zero), so the residual is exactly
    zero regardless of h0>0, mirroring the classical relativistic_fluid
    constant-state test."""
    residual = build_residual("quantum_relativistic_fluid", backend, speed_of_light=1.0, mass=1.0, hbar=1.0)
    pts = _points(6, 2, seed=74)

    def model(x):
        h = 1.3 * torch.ones_like(x[..., 0])
        u = torch.zeros_like(x[..., 0])
        return backend.stack([h, u], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)
    assert torch.allclose(r, torch.zeros_like(r), atol=_ATOL)


def test_einstein_field_minkowski_vacuum_solution():
    """Flat spacetime (g = diag(-1,1,1,1), constant everywhere) is the
    trivial exact vacuum solution of the Einstein Field Equations:
    Riemann/Ricci both vanish identically for a constant metric, so the
    residual should be exactly zero for any (t,x,y,z)."""
    residual = build_residual("einstein_field", backend)
    pts = _points(4, 4, seed=75)

    def model(x):
        n = x.shape[0]
        g_tt = -torch.ones(n)
        g_xx = torch.ones(n)
        g_yy = torch.ones(n)
        g_zz = torch.ones(n)
        zero = torch.zeros(n)
        # order: tt, tx, ty, tz, xx, xy, xz, yy, yz, zz
        return backend.stack([g_tt, zero, zero, zero, g_xx, zero, zero, g_yy, zero, g_zz], axis=-1)

    r = residual(model, pts)
    _assert_finite_and_shaped(r, 4)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-3), f"max|r|={r.abs().max().item()}"


ALL_TIER3_KEYS = {
    "poisson", "laplace", "heat", "diffusion", "wave", "advection", "helmholtz", "klein_gordon",
    "eikonal", "kdv",
    "biharmonic", "biharmonic_steady", "darcy", "euler_tricomi", "burgers",
    "fokker_planck", "fokker_planck_2d", "schrodinger", "allen_cahn", "cahn_hilliard",
    "cahn_hilliard_2d", "kuramoto_sivashinsky", "kadomtsev_petviashvili",
    "porous_medium", "perona_malik", "black_scholes_2d", "heston_volatility",
    "regularised_long_wave", "viscous_wave",
    "radhakrishnan_kundu_lakshmanan", "complex_ginzburg_landau_2d",
    "nlse_2d", "nls", "drift_diffusion_poisson", "brinkman_darcy",
    "burgers_2d", "boussinesq_convection", "boussinesq_wave",
    "navier_stokes", "stokes", "sine_gordon_2d",
    "klein_gordon_nonlinear_2d", "reaction_diffusion",
    "fisher_kpp", "fitzhugh_nagumo", "gierer_meinhardt", "gray_scott",
    "swift_hohenberg", "phase_field_crystal", "dendritic_solidification",
    "relativistic_fluid", "euler", "shallow_water_2d",
    # Sprint additions
    "phonon", "phonons", "dirac", "dirac_equation",
    "bose_einstein", "gross_pitaevskii", "fermi_gas", "fermi_dirac",
    "quantum_relativistic_fluid", "einstein_field", "efe",
}


def test_tier3_covers_the_entire_registry():
    """Guards against silently losing coverage: every registered PDE (plus
    eikonal) must have a Tier-3 (exact/manufactured) or Tier-3b
    (constant-equilibrium) test above — this is the "all PDEs, not just 8"
    expansion of the original curated Tier-3 set."""
    missing = set(ALL_REGISTRY_KEYS) - set(ALL_TIER3_KEYS)
    assert not missing, f"PDEs with no Tier-3/3b manufactured-solution test: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Registry / factory behavior
# ---------------------------------------------------------------------------

def test_registry_aliases_point_to_the_same_class():
    assert PDE_REGISTRY["laplace"] is PDE_REGISTRY["poisson"]
    assert PDE_REGISTRY["diffusion"] is PDE_REGISTRY["heat"]


def test_registry_has_no_duplicate_unintended_aliases():
    """Every key other than the two documented aliases should map to a
    distinct class — guards against a copy-paste registry entry silently
    shadowing a different equation."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for key, cls in PDE_REGISTRY.items():
        by_class[cls].append(key)
    unexpected_dupes = {
        cls: keys for cls, keys in by_class.items()
        if len(keys) > 1 and set(keys) not in (
            {"poisson", "laplace"},
            {"heat", "diffusion"},
            {"einstein_field", "efe"},
            {"phonon", "phonons"},
            {"dirac", "dirac_equation"},
            {"bose_einstein", "gross_pitaevskii"},
            {"fermi_gas", "fermi_dirac"},
        )
    }
    assert not unexpected_dupes, f"Unexpected duplicate registry entries: {unexpected_dupes}"


def test_build_residual_unknown_name_raises():
    with pytest.raises(ValueError):
        build_residual("not_a_real_pde", backend)


def test_build_residual_error_lists_available_names():
    with pytest.raises(ValueError, match="poisson"):
        build_residual("not_a_real_pde", backend)


def test_advection_requires_velocity_kwarg():
    with pytest.raises(TypeError):
        build_residual("advection", backend)


# ---------------------------------------------------------------------------
# atan availability across backends
# ---------------------------------------------------------------------------

def test_dendritic_solidification_does_not_crash_on_missing_atan():
    """DendriticSolidificationResidual calls backend.atan(), which must be
    available on every backend (Torch/JAX/TensorFlow/Dedalus) for this PDE
    to be usable."""
    residual = build_residual("dendritic_solidification", backend)
    pts = _points(6, 3, seed=10)
    model = _smooth_nonzero_model(2)
    r = residual(model, pts)
    _assert_finite_and_shaped(r, 6)


def test_backend_atan_matches_torch_atan():
    x = backend.tensor(np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32))
    assert torch.allclose(backend.atan(x), torch.atan(x))