"""
physai/core/pde_residual.py

Backend-agnostic PDE residual evaluators.

Each class implements a ``__call__(model_fn, points) -> Tensor`` interface
that returns the strong-form residual at collocation points.
All differentiation is performed through the injected AbstractBackend,
so the same residual code runs on PyTorch, JAX, and TensorFlow.

Equations implemented
---------------------
 1. Poisson / Laplace                (elliptic)
 2. Heat / Diffusion                 (parabolic)
 3. Wave                             (hyperbolic)
 4. Burgers                          (nonlinear, 1-D)
 5. Navier-Stokes (incompressible)   (coupled PDE system)
 6. Advection                        (linear transport)
 7. Helmholtz                        (elliptic, frequency-domain)
 8. Allen-Cahn                       (phase field)
 9. Cahn-Hilliard                    (4th-order phase field)
10. Schrödinger (time-dependent)     (quantum)
11. Klein-Gordon                     (relativistic scalar)
12. Reaction-Diffusion               (Turing patterns)
13. Eikonal                          (ray tracing / level sets)
14. Darcy flow                       (porous media)
15. Euler equations (compressible)   (inviscid gas dynamics)
16. Maxwell (curl-curl form)         (electromagnetics)
17. Biharmonic                       (plate bending)
18. Stokes                           (creeping flow)
19. Nonlinear Schrödinger (NLS)      (nonlinear optics / BEC)
20. Korteweg-de Vries (KdV)          (dispersive waves)
21. Fokker-Planck                    (probability density)
41. Einstein Field Equations         (general relativity, full nonlinear 4-D)
42. Phonons                          (dispersive lattice wave)
43. Dirac Equation (1+1-D)           (relativistic quantum spinor)
44. Bose-Einstein Condensate (GPE)   (quantum gas, trapped)
45. Fermi Gas (degenerate)           (quantum gas, Fermi-Dirac closure)
46. Quantum Relativistic Fluid       (relativistic fluid + Bohm potential)
"""
from __future__ import annotations

import difflib
import inspect
import math
import re
import threading
from dataclasses import replace as _dc_replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from physai.backends.base import AbstractBackend, Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grad(backend: AbstractBackend, fn: Callable, inputs: Tensor, *, mode: str = "reverse") -> Tensor:
    """First-order gradient of scalar ``fn`` w.r.t. ``inputs``.

    ``mode`` is forwarded to ``backend.grad`` — see ``AbstractBackend.grad``
    for the reverse/forward/taylor tradeoffs. Reverse mode remains the
    right default for the common case (scalar output, many collocation
    points); forward mode pays off for wide coupled/mixed residual outputs.
    """
    return backend.grad(fn, mode=mode)(inputs)


def _grad2(backend: AbstractBackend, fn: Callable, inputs: Tensor, *, mode: str = "reverse") -> Tensor:
    """Second-order gradient (Hessian diagonal sum → Laplacian helper)."""
    if mode == "taylor":
        # Taylor-mode already returns second directional derivatives per
        # input dimension in one forward sweep — no need to nest grads.
        return backend.grad(fn, mode="taylor")(inputs)
    g  = backend.grad(fn, mode=mode)
    gg = backend.grad(lambda x: backend.sum(g(x)), mode=mode)
    return gg(inputs)


def _laplacian(
    backend: AbstractBackend,
    fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    spatial_dims: Optional[int] = None,
    *,
    mode: str = "reverse",
) -> Tensor:
    """
    Compute the Laplacian Δu = Σ_i ∂²u/∂x_i² at ``inputs``.

    Parameters
    ----------
    spatial_dims : number of leading columns of ``inputs`` that are spatial.
        Defaults to ``inputs.shape[-1]`` (all columns). For time-dependent
        residuals where the last column of ``inputs`` is time, callers MUST
        pass ``spatial_dims=inputs.shape[-1] - 1`` so the time column is not
        folded into the spatial Laplacian.
    mode : ``"reverse"`` (default, nested reverse-mode grads — universally
        safe), ``"forward"`` (per-dimension JVP sweep, cheaper for wide
        outputs), or ``"taylor"`` (forward-over-forward / jet: one forward
        sweep per input dimension gets the second directional derivative
        directly, which is the cheapest option when spatial_dims is small,
        e.g. 2D/3D PDEs — the common case).
    """
    n_dims = int(inputs.shape[-1]) if spatial_dims is None else spatial_dims

    if mode == "taylor":
        def u_sum(x: Tensor) -> Tensor:
            return backend.sum(fn(x))

        second = backend.grad(u_sum, mode="taylor")(inputs)  # (..., n_dims_full) second-derivs
        lap = backend.zeros(inputs.shape[:-1])
        for i in range(n_dims):
            lap = lap + second[..., i]
        return lap

    # First-order gradient function (summed to output a scalar for backend.grad)
    def u_sum(x: Tensor) -> Tensor:
        return backend.sum(fn(x))

    g_fn = backend.grad(u_sum, mode=mode)
    lap  = backend.zeros(inputs.shape[:-1])

    # Sequential per-component Hessian diagonal (universally safe)
    for i in range(n_dims):
        def _gi(x: Tensor, _i: int = i) -> Tensor:
            g = g_fn(x)
            # Basic slicing [..., _i] isolates the component backend-agnostically.
            # Summing it ensures a scalar output for the outer grad function.
            return backend.sum(g[..., _i])

        # backend.grad(_gi)(inputs) returns a tensor of shape (..., n_dims).
        # We slice out the i-th column to get the pure second derivative (∂²u / ∂x_i²)
        lap = lap + backend.grad(_gi, mode=mode)(inputs)[..., i]

    return lap

def _time_deriv(
    backend: AbstractBackend,
    fn: Callable[[Tensor], Tensor],
    xt: Tensor,
    t_index: int = -1,
    *,
    mode: str = "reverse",
) -> Tensor:
    """
    Partial derivative of ``fn`` w.r.t. the coordinate at ``t_index``.
    By convention the last column of ``xt`` is time.
    """
    g = backend.grad(lambda x: backend.sum(fn(x)), mode=mode)(xt)
    # Extract column t_index
    return g[..., t_index]


def _spatial_grad(
    backend: AbstractBackend,
    fn: Callable[[Tensor], Tensor],
    xt: Tensor,
    space_axes: Optional[Sequence[int]] = None,
    *,
    mode: str = "reverse",
) -> Tensor:
    """
    Gradient of ``fn`` w.r.t. spatial coordinates only.
    If ``space_axes`` is None, all axes except the last (time) are spatial.
    """
    g = backend.grad(lambda x: backend.sum(fn(x)), mode=mode)(xt)
    if space_axes is None:
        return g[..., :-1]
    idx = list(space_axes)
    # Gather spatial columns
    cols = [g[..., i : i + 1] for i in idx]
    return backend.concatenate(cols, axis=-1)


# ---------------------------------------------------------------------------
# Base residual class
# ---------------------------------------------------------------------------

class PDEResidual:
    """Base class for all PDE residual evaluators.

    Parameters
    ----------
    diff_mode : differentiation strategy passed through to every
        ``backend.grad`` call made while evaluating this residual — one of
        ``"reverse"`` (default), ``"forward"``, or ``"taylor"``. See
        ``AbstractBackend.grad`` for the tradeoffs. Individual residual
        subclasses that call the module-level ``_grad``/``_grad2``/
        ``_laplacian``/``_time_deriv``/``_spatial_grad`` helpers should pass
        ``mode=self.diff_mode`` through so the setting takes effect.
    """

    def __init__(
        self,
        backend: AbstractBackend,
        *,
        diff_mode: str = "reverse",
        **params: Any,
    ) -> None:
        self.backend    = backend
        self.diff_mode  = diff_mode
        self.params     = params

    def __call__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def loss(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Tensor:
        """Convenience: MSE of the residual."""
        r = self(model_fn, points)
        return self.backend.mean(self.backend.square(r))


# ---------------------------------------------------------------------------
# 1. Poisson / Laplace
# ---------------------------------------------------------------------------

class PoissonResidual(PDEResidual):
    """
    -Δu = f(x)   (Poisson)
    -Δu = 0      (Laplace when f ≡ 0)

    Parameters
    ----------
    source_fn : optional callable f(x) → Tensor; default zero
    """

    def __init__(
        self,
        backend: AbstractBackend,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.source_fn = source_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b   = self.backend
        lap = _laplacian(b, model_fn, points, mode=self.diff_mode)
        f   = self.source_fn(points)
        return -lap - f


# ---------------------------------------------------------------------------
# 2. Heat / Diffusion
# ---------------------------------------------------------------------------

class HeatResidual(PDEResidual):
    """
    ∂u/∂t - α Δu = f(x, t)

    ``points`` columns: [x₁, …, x_d, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        alpha: float = 1.0,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, alpha=alpha, diff_mode=diff_mode)
        self.alpha     = alpha
        self.source_fn = source_fn or (lambda xt: backend.zeros(xt.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend

        u_t = _time_deriv(b, model_fn, points, t_index=-1, mode=self.diff_mode)
        lap = _laplacian(
            b, lambda x: model_fn(x), points,
            spatial_dims=points.shape[-1] - 1, mode=self.diff_mode,
        )
        f   = self.source_fn(points)
        return u_t - self.alpha * lap - f


# ---------------------------------------------------------------------------
# 3. Wave equation
# ---------------------------------------------------------------------------

class WaveResidual(PDEResidual):
    """
    ∂²u/∂t² - c² Δu = f(x, t)

    ``points`` columns: [x₁, …, x_d, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        c: float = 1.0,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, c=c, diff_mode=diff_mode)
        self.c         = c
        self.source_fn = source_fn or (lambda xt: backend.zeros(xt.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        # ∂u/∂t
        u_t_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_t    = u_t_fn(points)[..., -1]

        # ∂²u/∂t²
        u_tt_fn = b.grad(lambda x: b.sum(u_t_fn(x)[..., -1]), mode=self.diff_mode)
        u_tt    = u_tt_fn(points)[..., -1]

        lap = _laplacian(b, model_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        f   = self.source_fn(points)
        return u_tt - self.c ** 2 * lap - f


# ---------------------------------------------------------------------------
# 4. Burgers equation (1-D)
# ---------------------------------------------------------------------------

class BurgersResidual(PDEResidual):
    """
    ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0

    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        nu: float = 0.01 / 3.14159265,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, nu=nu, diff_mode=diff_mode)
        self.nu = nu

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        g_fn  = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        gg_fn = b.grad(lambda x: b.sum(g_fn(x)[..., 0]), mode=self.diff_mode)  # ∂²u/∂x²

        g   = g_fn(points)     # [N, 2]: [∂u/∂x, ∂u/∂t]
        u_x = g[..., 0]
        u_t = g[..., 1]
        u   = model_fn(points)[..., 0]

        u_xx = gg_fn(points)[..., 0]

        return u_t + u * u_x - self.nu * u_xx


# ---------------------------------------------------------------------------
# 5. Navier-Stokes (incompressible, 2-D)
# ---------------------------------------------------------------------------

class NavierStokesResidual(PDEResidual):
    """
    Incompressible Navier-Stokes in 2-D:
        ρ (∂u/∂t + u·∇u) = -∇p + μ Δu
        ∇·u = 0

    ``model_fn(xt)`` → [u, v, p]  (3 outputs)
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        rho: float = 1.0,
        mu: float = 0.01,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, rho=rho, mu=mu, diff_mode=diff_mode)
        self.rho = rho
        self.mu  = mu

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        def _component(fn: Callable, idx: int) -> Callable:
            return lambda x: fn(x)[..., idx : idx + 1]

        u_fn = lambda x: model_fn(x)[..., 0:1]  # noqa: E731
        v_fn = lambda x: model_fn(x)[..., 1:2]  # noqa: E731
        p_fn = lambda x: model_fn(x)[..., 2:3]  # noqa: E731

        g_u = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)  # [∂u/∂x, ∂u/∂y, ∂u/∂t]
        g_v = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)
        g_p = b.grad(lambda x: b.sum(p_fn(x)), mode=self.diff_mode)(points)

        u, v = model_fn(points)[..., 0], model_fn(points)[..., 1]

        u_x, u_y, u_t = g_u[..., 0], g_u[..., 1], g_u[..., 2]
        v_x, v_y, v_t = g_v[..., 0], g_v[..., 1], g_v[..., 2]
        p_x, p_y       = g_p[..., 0], g_p[..., 1]

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        momentum_x = self.rho * (u_t + u * u_x + v * u_y) + p_x - self.mu * lap_u
        momentum_y = self.rho * (v_t + u * v_x + v * v_y) + p_y - self.mu * lap_v
        continuity  = u_x + v_y

        return b.stack([momentum_x, momentum_y, continuity], axis=-1)


# ---------------------------------------------------------------------------
# 6. Advection
# ---------------------------------------------------------------------------

class AdvectionResidual(PDEResidual):
    """
    ∂u/∂t + c · ∇u = 0

    ``points`` columns: [x₁, …, x_d, t]
    ``velocity`` : constant advection speed vector of length d
    """

    def __init__(
        self,
        backend: AbstractBackend,
        velocity: Sequence[float],
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.velocity = velocity

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend
        g  = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)(points)

        u_t  = g[..., -1]
        u_x  = g[..., :-1]
        c    = b.tensor(list(self.velocity))
        conv = b.sum(u_x * c, axis=-1)
        return u_t + conv


# ---------------------------------------------------------------------------
# 7. Helmholtz
# ---------------------------------------------------------------------------

class HelmholtzResidual(PDEResidual):
    """
    Δu + k² u = f(x)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        k: float = 1.0,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, k=k, diff_mode=diff_mode)
        self.k         = k
        self.source_fn = source_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b   = self.backend
        u   = model_fn(points)[..., 0]
        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, mode=self.diff_mode)
        f   = self.source_fn(points)
        return lap + self.k ** 2 * u - f


# ---------------------------------------------------------------------------
# 8. Allen-Cahn
# ---------------------------------------------------------------------------

class AllenCahnResidual(PDEResidual):
    """
    ∂u/∂t = ε² Δu + u(1 - u²)

    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        epsilon: float = 0.01,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, epsilon=epsilon, diff_mode=diff_mode)
        self.epsilon = epsilon

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b   = self.backend
        u   = model_fn(points)[..., 0]
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        return u_t - self.epsilon ** 2 * lap - u * (1.0 - u ** 2)


# ---------------------------------------------------------------------------
# 9. Cahn-Hilliard (4th order)
# ---------------------------------------------------------------------------

class CahnHilliardResidual(PDEResidual):
    """
    ∂u/∂t = Δ(-ε² Δu + W'(u))    W'(u) = u³ - u

    Implemented as a coupled system:
        w = -ε² Δu + W'(u)
        ∂u/∂t = Δw
    """

    def __init__(
        self,
        backend: AbstractBackend,
        epsilon: float = 0.01,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, epsilon=epsilon, diff_mode=diff_mode)
        self.epsilon = epsilon

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b   = self.backend
        eps = self.epsilon

        # model_fn → [u, w]
        u_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731
        w_fn = lambda x: model_fn(x)[..., 1]  # noqa: E731

        u    = u_fn(points)
        u_t  = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap_u = _laplacian(b, u_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_w = _laplacian(b, w_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        Wp   = u ** 3 - u

        residual_u = u_t - lap_w
        residual_w = w_fn(points) - (-eps ** 2 * lap_u + Wp)

        return b.stack([residual_u, residual_w], axis=-1)


# ---------------------------------------------------------------------------
# 10. Time-dependent Schrödinger
# ---------------------------------------------------------------------------

class SchrodingerResidual(PDEResidual):
    """
    iℏ ∂ψ/∂t = -ℏ²/(2m) Δψ + V(x) ψ

    model_fn → [Re(ψ), Im(ψ)]
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        hbar: float = 1.0,
        mass: float = 1.0,
        potential_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.hbar        = hbar
        self.mass        = mass
        self.potential_fn = potential_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend
        h  = self.hbar
        m  = self.mass

        psi_r_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731
        psi_i_fn = lambda x: model_fn(x)[..., 1]  # noqa: E731

        psi_r = psi_r_fn(points)
        psi_i = psi_i_fn(points)

        psi_r_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        psi_i_t = _time_deriv(b, lambda x: model_fn(x)[..., 1:2], points, mode=self.diff_mode)

        lap_r = _laplacian(b, psi_r_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_i = _laplacian(b, psi_i_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        V = self.potential_fn(points)

        coeff = -(h ** 2) / (2.0 * m)

        # iℏ (ψ_r_t + i ψ_i_t) = (coeff Δ + V)(ψ_r + i ψ_i)
        # Real part: -ℏ ψ_i_t = coeff Δψ_r + V ψ_r
        # Imag part:  ℏ ψ_r_t = coeff Δψ_i + V ψ_i
        res_r = -h * psi_i_t - coeff * lap_r - V * psi_r
        res_i =  h * psi_r_t - coeff * lap_i - V * psi_i

        return b.stack([res_r, res_i], axis=-1)


# ---------------------------------------------------------------------------
# 11. Klein-Gordon
# ---------------------------------------------------------------------------

class KleinGordonResidual(PDEResidual):
    """
    ∂²u/∂t² - c² Δu + m² u = f(x, t)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        c: float = 1.0,
        m: float = 1.0,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, c=c, m=m, diff_mode=diff_mode)
        self.c         = c
        self.m         = m
        self.source_fn = source_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        g_fn  = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_t   = g_fn(points)[..., -1]
        u_tt  = b.grad(lambda x: b.sum(g_fn(x)[..., -1]), mode=self.diff_mode)(points)[..., -1]

        u   = model_fn(points)[..., 0]
        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        f   = self.source_fn(points)

        return u_tt - self.c ** 2 * lap + self.m ** 2 * u - f


# ---------------------------------------------------------------------------
# 12. Reaction-Diffusion
# ---------------------------------------------------------------------------

class ReactionDiffusionResidual(PDEResidual):
    """
    ∂u/∂t = D Δu + R(u)    (single species)

    Parameters
    ----------
    D         : diffusion coefficient
    reaction_fn : R(u) → Tensor  (default: logistic u(1-u))
    """

    def __init__(
        self,
        backend: AbstractBackend,
        D: float = 0.1,
        reaction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, D=D, diff_mode=diff_mode)
        self.D           = D
        self.reaction_fn = reaction_fn or (lambda u: u * (1.0 - u))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b   = self.backend
        u   = model_fn(points)[..., 0]
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        R   = self.reaction_fn(u)
        return u_t - self.D * lap - R


# ---------------------------------------------------------------------------
# 13. Eikonal
# ---------------------------------------------------------------------------

class EikonalResidual(PDEResidual):
    """
    |∇u| = f(x)

    ``points`` columns: [x₁, …, x_d]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        speed_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.speed_fn = speed_fn or (lambda x: backend.ones(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b    = self.backend
        g    = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)(points)
        norm = b.sqrt(b.sum(b.square(g), axis=-1) + 1e-8)
        f    = self.speed_fn(points)
        return norm - f


# ---------------------------------------------------------------------------
# 14. Darcy Flow
# ---------------------------------------------------------------------------

class DarcyResidual(PDEResidual):
    """
    -∇·(K(x) ∇u) = f(x)

    Parameters
    ----------
    permeability_fn : K(x) → scalar Tensor (default: 1)
    source_fn       : f(x) → scalar Tensor (default: 0)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        permeability_fn: Optional[Callable[[Tensor], Tensor]] = None,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.K_fn = permeability_fn or (lambda x: backend.ones(x.shape[:-1]))
        self.f_fn = source_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        # ∇·(K ∇u) via summing ∂(K ∇u_i)/∂x_i
        # Approximate via divergence of a vector field through another grad call
        def Kgrad_sum(x: Tensor) -> Tensor:
            g = b.grad(lambda y: b.sum(model_fn(y)), mode=self.diff_mode)(x)
            k = self.K_fn(x)
            return k[..., None] * g

        div_Kgrad = b.zeros(points.shape[:-1])
        n_dims    = int(points.shape[-1])
        for i in range(n_dims):
            def _Kgradi(x: Tensor, _i: int = i) -> Tensor:
                return Kgrad_sum(x)[..., _i]
            div_Kgrad = div_Kgrad + b.grad(lambda x, _i=i: b.sum(_Kgradi(x)), mode=self.diff_mode)(points)[..., i]

        f = self.f_fn(points)
        return -div_Kgrad - f


# ---------------------------------------------------------------------------
# 15. Compressible Euler (1-D)
# ---------------------------------------------------------------------------

class EulerResidual(PDEResidual):
    """
    1-D compressible Euler in conservative form:
        ∂ρ/∂t + ∂(ρu)/∂x = 0
        ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
        ∂E/∂t + ∂(u(E + p))/∂x = 0
        p = (γ-1)(E - ½ρu²)

    model_fn → [ρ, u, E]   ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        gamma: float = 1.4,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, gamma=gamma, diff_mode=diff_mode)
        self.gamma = gamma

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend
        g  = self.gamma

        out  = model_fn(points)
        rho, u, E = out[..., 0], out[..., 1], out[..., 2]
        p    = (g - 1.0) * (E - 0.5 * rho * u ** 2)

        def _t(field_fn: Callable) -> Tensor:
            return b.grad(lambda x: b.sum(field_fn(x)), mode=self.diff_mode)(points)[..., -1]

        def _x(field_fn: Callable) -> Tensor:
            return b.grad(lambda x: b.sum(field_fn(x)), mode=self.diff_mode)(points)[..., 0]

        rho_t  = _t(lambda x: model_fn(x)[..., 0:1])
        rhou_fn = lambda x: model_fn(x)[..., 0] * model_fn(x)[..., 1]  # noqa: E731
        E_fn    = lambda x: model_fn(x)[..., 2]                          # noqa: E731

        cont   = rho_t + _x(rhou_fn)

        rhou2p = lambda x: (  # noqa: E731
            model_fn(x)[..., 0] * model_fn(x)[..., 1] ** 2
            + (g - 1.0) * (model_fn(x)[..., 2] - 0.5 * model_fn(x)[..., 0] * model_fn(x)[..., 1] ** 2)
        )
        rhou_t = _t(lambda x: model_fn(x)[..., 0:1] * model_fn(x)[..., 1:2])
        mom    = rhou_t + _x(rhou2p)

        uEp_fn = lambda x: (  # noqa: E731
            model_fn(x)[..., 1]
            * (model_fn(x)[..., 2]
               + (g - 1.0) * (model_fn(x)[..., 2] - 0.5 * model_fn(x)[..., 0] * model_fn(x)[..., 1] ** 2))
        )
        E_t    = _t(E_fn)
        energy = E_t + _x(uEp_fn)

        return b.stack([cont, mom, energy], axis=-1)


# ---------------------------------------------------------------------------
# 16. Biharmonic
# ---------------------------------------------------------------------------

class BiharmonicResidual(PDEResidual):
    """
    Δ²u = f(x)   (plate bending)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.source_fn = source_fn or (lambda x: backend.zeros(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend
        # Δ(Δu)
        lap_u  = lambda x: _laplacian(b, model_fn, x, mode=self.diff_mode)  # noqa: E731
        lap2_u = _laplacian(b, lap_u, points, mode=self.diff_mode)
        return lap2_u - self.source_fn(points)


# ---------------------------------------------------------------------------
# 17. Stokes (2-D)
# ---------------------------------------------------------------------------

class StokesResidual(PDEResidual):
    """
    Steady Stokes (creeping flow), 2-D:
        -μ Δu + ∇p = f
        ∇·u = 0

    model_fn → [u, v, p]   ``points`` columns: [x, y]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mu: float = 1.0,
        body_force: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, mu=mu, diff_mode=diff_mode)
        self.mu         = mu
        self.body_force = body_force or (lambda x: backend.zeros((*x.shape[:-1], 2)))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        u_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731
        v_fn = lambda x: model_fn(x)[..., 1]  # noqa: E731
        p_fn = lambda x: model_fn(x)[..., 2]  # noqa: E731

        lap_u = _laplacian(b, u_fn, points, mode=self.diff_mode)
        lap_v = _laplacian(b, v_fn, points, mode=self.diff_mode)

        g_p   = b.grad(lambda x: b.sum(p_fn(x)), mode=self.diff_mode)(points)
        p_x   = g_p[..., 0]
        p_y   = g_p[..., 1]

        g_u   = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)
        g_v   = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)
        div   = g_u[..., 0] + g_v[..., 1]

        f = self.body_force(points)
        mom_x = -self.mu * lap_u + p_x - f[..., 0]
        mom_y = -self.mu * lap_v + p_y - f[..., 1]

        return b.stack([mom_x, mom_y, div], axis=-1)


# ---------------------------------------------------------------------------
# 18. Nonlinear Schrödinger (NLS / Gross-Pitaevskii)
# ---------------------------------------------------------------------------

class NLSResidual(PDEResidual):
    """
    iℏ ∂ψ/∂t = -ℏ²/(2m) Δψ + g|ψ|²ψ

    model_fn → [Re(ψ), Im(ψ)]   ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        hbar: float = 1.0,
        mass: float = 1.0,
        g: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.hbar = hbar
        self.mass = mass
        self.g    = g

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b  = self.backend
        h, m, g = self.hbar, self.mass, self.g

        psi_r = lambda x: model_fn(x)[..., 0]  # noqa: E731
        psi_i = lambda x: model_fn(x)[..., 1]  # noqa: E731

        r  = psi_r(points)
        i  = psi_i(points)
        n2 = r ** 2 + i ** 2   # |ψ|²

        r_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        i_t = _time_deriv(b, lambda x: model_fn(x)[..., 1:2], points, mode=self.diff_mode)

        lap_r = _laplacian(b, psi_r, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_i = _laplacian(b, psi_i, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        coeff = -(h ** 2) / (2.0 * m)

        res_r = -h * i_t - coeff * lap_r - g * n2 * r
        res_i =  h * r_t - coeff * lap_i - g * n2 * i

        return b.stack([res_r, res_i], axis=-1)


# ---------------------------------------------------------------------------
# 19. KdV
# ---------------------------------------------------------------------------

class KdVResidual(PDEResidual):
    """
    ∂u/∂t + 6u ∂u/∂x + ∂³u/∂x³ = 0

    ``points`` columns: [x, t]
    """

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        g_fn   = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        gg_fn  = b.grad(lambda x: b.sum(g_fn(x)[..., 0]), mode=self.diff_mode)
        ggg_fn = b.grad(lambda x: b.sum(gg_fn(x)[..., 0]), mode=self.diff_mode)

        g    = g_fn(points)
        u    = model_fn(points)[..., 0]
        u_x  = g[..., 0]
        u_t  = g[..., 1]
        u_xxx = ggg_fn(points)[..., 0]

        return u_t + 6.0 * u * u_x + u_xxx


# ---------------------------------------------------------------------------
# 20. Fokker-Planck
# ---------------------------------------------------------------------------

class FokkerPlanckResidual(PDEResidual):
    """
    ∂p/∂t = -∇·(A(x)p) + ½ ∇²(D(x)p)

    Parameters
    ----------
    drift_fn     : A(x) → Tensor[..., d]   (default: 0)
    diffusion_fn : D(x) → scalar Tensor    (default: 1)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        drift_fn: Optional[Callable[[Tensor], Tensor]] = None,
        diffusion_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.drift_fn     = drift_fn     or (lambda x: backend.zeros(x.shape))
        self.diffusion_fn = diffusion_fn or (lambda x: backend.ones(x.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b    = self.backend
        p_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731

        p   = p_fn(points)
        A   = self.drift_fn(points[..., :-1])    # spatial only
        D   = self.diffusion_fn(points[..., :-1])

        p_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)

        # Divergence of (A * p)
        n_dims = int(points.shape[-1]) - 1  # spatial dims
        div_Ap  = b.zeros(points.shape[:-1])
        for i in range(n_dims):
            def _Api(x: Tensor, _i: int = i) -> Tensor:
                p_v  = model_fn(x)[..., 0]
                a_v  = self.drift_fn(x[..., :-1])[..., _i]
                return a_v * p_v
            div_Ap = div_Ap + b.grad(lambda x, _i=i: b.sum(_Api(x)), mode=self.diff_mode)(points)[..., i]

        # Laplacian of (D * p)
        Dp_fn  = lambda x: self.diffusion_fn(x[..., :-1]) * model_fn(x)[..., 0]  # noqa: E731
        lap_Dp = _laplacian(b, Dp_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        return p_t + div_Ap - 0.5 * lap_Dp

# ---------------------------------------------------------------------------
# 9. 2D Nonlinear Schrödinger Equation (NLSE)
# ---------------------------------------------------------------------------

class NonlinearSchrodinger2DResidual(PDEResidual):
    """
    i ∂ψ/∂t + Δψ + β |ψ|² ψ = 0
    
    Split into real components where ψ = u + iv:
        ∂u/∂t = -Δv - β (u² + v²) v
        ∂v/∂t =  Δu + β (u² + v²) u

    ``model_fn(xyt)`` → [u, v] (2 outputs)
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        beta: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, beta=beta, diff_mode=diff_mode)
        self.beta = beta

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, t_index=-1, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, t_index=-1, mode=self.diff_mode)

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        u = model_fn(points)[..., 0]
        v = model_fn(points)[..., 1]
        mag_sq = u**2 + v**2

        # res_u corresponds to the real-part equation (du/dt = -lap(v) - beta*|psi|^2*v)
        # res_v corresponds to the imaginary-part equation (dv/dt = lap(u) + beta*|psi|^2*u)
        res_u = u_t + lap_v + self.beta * mag_sq * v
        res_v = v_t - lap_u - self.beta * mag_sq * u

        return b.stack([res_u, res_v], axis=-1)


# ---------------------------------------------------------------------------
# 10. Sine-Gordon Equation (2D)
# ---------------------------------------------------------------------------

class SineGordon2DResidual(PDEResidual):
    """
    ∂²u/∂t² - Δu + sin(u) = 0
    
    ``points`` columns: [x, y, t]
    """

    def __init__(self, backend: AbstractBackend, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, diff_mode=diff_mode)

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]

        u_t_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_t    = u_t_fn(points)[..., -1]
        
        u_tt_fn = b.grad(lambda x: b.sum(u_t_fn(x)[..., -1]), mode=self.diff_mode)
        u_tt    = u_tt_fn(points)[..., -1]

        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        return u_tt - lap + b.sin(u)


# ---------------------------------------------------------------------------
# 12. Fisher's Kolmogorov-Petrovsky-Piskunov (KPP) Equation (2D)
# ---------------------------------------------------------------------------

class FisherKPPResidual(PDEResidual):
    """
    ∂u/∂t - D Δu - r u (1 - u) = 0
    
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self, 
        backend: AbstractBackend, 
        D: float = 0.1, 
        r: float = 1.0, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, D=D, r=r, diff_mode=diff_mode)
        self.D = D
        self.r = r

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        
        return u_t - self.D * lap - self.r * u * (1.0 - u)


# ---------------------------------------------------------------------------
# 13. Klein-Gordon Equation (2D)
# ---------------------------------------------------------------------------

class KleinGordonNonlinear2DResidual(PDEResidual):
    """
    ∂²u/∂t² - Δu + m² u + γ u³ = 0
    
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self, 
        backend: AbstractBackend, 
        m: float = 1.0, 
        gamma: float = 0.0, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, m=m, gamma=gamma, diff_mode=diff_mode)
        self.m = m
        self.gamma = gamma

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]

        u_t_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_tt_fn = b.grad(lambda x: b.sum(u_t_fn(x)[..., -1]), mode=self.diff_mode)
        u_tt = u_tt_fn(points)[..., -1]

        lap = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        return u_tt - lap + (self.m**2) * u + self.gamma * (u**3)


# ---------------------------------------------------------------------------
# 14. FitzHugh-Nagumo System
# ---------------------------------------------------------------------------

class FitzHughNagumoResidual(PDEResidual):
    """
    ∂v/∂t = D_v Δv + v - v³ - w + I
    ∂w/∂t = D_w Δw + a v - b w
    
    ``model_fn(xt)`` → [v, w]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        D_v: float = 1.0,
        D_w: float = 0.0,
        a: float = 0.5,
        b: float = 0.8,
        I: float = 0.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, D_v=D_v, D_w=D_w, a=a, b=b, I=I, diff_mode=diff_mode)
        self.D_v, self.D_w = D_v, D_w
        self.a, self.b, self.I = a, b, I

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        v_fn = lambda x: model_fn(x)[..., 0:1]
        w_fn = lambda x: model_fn(x)[..., 1:2]

        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)
        w_t = _time_deriv(b, lambda x: w_fn(x)[..., 0], points, mode=self.diff_mode)

        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_w = _laplacian(b, lambda x: w_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        v = model_fn(points)[..., 0]
        w = model_fn(points)[..., 1]

        res_v = v_t - self.D_v * lap_v - (v - v**3 - w + self.I)
        res_w = w_t - self.D_w * lap_w - (self.a * v - self.b * w)
        return b.stack([res_v, res_w], axis=-1)


# ---------------------------------------------------------------------------
# 15. Cahn-Hilliard Equation (4th-order Split Formulation)
# ---------------------------------------------------------------------------

class CahnHilliard2DResidual(PDEResidual):
    """
    ∂c/∂t - Δmu = 0
    mu = c³ - c - γ Δc
        
    ``model_fn(xt)`` → [c, mu]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        gamma: float = 0.01, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, gamma=gamma, diff_mode=diff_mode)
        self.gamma = gamma

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        c_fn = lambda x: model_fn(x)[..., 0:1]
        mu_fn = lambda x: model_fn(x)[..., 1:2]

        c = model_fn(points)[..., 0]
        mu = model_fn(points)[..., 1]

        n_dims = points.shape[-1] - 1
        c_t = _time_deriv(b, lambda x: c_fn(x)[..., 0], points, mode=self.diff_mode)
        lap_c = _laplacian(b, lambda x: c_fn(x)[..., 0], points, spatial_dims=n_dims, mode=self.diff_mode)
        lap_mu = _laplacian(b, lambda x: mu_fn(x)[..., 0], points, spatial_dims=n_dims, mode=self.diff_mode)

        res_mu = mu - (c**3 - c - self.gamma * lap_c)
        res_c  = c_t - lap_mu
        return b.stack([res_c, res_mu], axis=-1)


# ---------------------------------------------------------------------------
# 16. Kuramoto-Sivashinsky (KS) Equation (1D)
# ---------------------------------------------------------------------------

class KuramotoSivashinskyResidual(PDEResidual):
    """
    ∂u/∂t + u ∂u/∂x + α ∂²u/∂x² + β ∂⁴u/∂x⁴ = 0
    
    ``points`` columns: [x, t]
    """

    def __init__(
        self, 
        backend: AbstractBackend, 
        alpha: float = 1.0, 
        beta: float = 1.0, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, alpha=alpha, beta=beta, diff_mode=diff_mode)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]

        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_x = g1_fn(points)[..., 0]
        u_t = g1_fn(points)[..., 1]

        g2_fn = b.grad(lambda x: b.sum(g1_fn(x)[..., 0]), mode=self.diff_mode)
        u_xx = g2_fn(points)[..., 0]

        g3_fn = b.grad(lambda x: b.sum(g2_fn(x)[..., 0]), mode=self.diff_mode)
        g4_fn = b.grad(lambda x: b.sum(g3_fn(x)[..., 0]), mode=self.diff_mode)
        u_xxxx = g4_fn(points)[..., 0]

        return u_t + u * u_x + self.alpha * u_xx + self.beta * u_xxxx

# ---------------------------------------------------------------------------
# 17. Kadomtsev-Petviashvili (KP) Equation (2D Dispersive Wave)
# ---------------------------------------------------------------------------

class KadomtsevPetviashviliResidual(PDEResidual):
    """
    ∂/∂x (∂u/∂t + 6u ∂u/∂x + ∂³u/∂x³) + λ ∂²u/∂y² = 0
    
    KP-I when λ = -1 (strong surface tension), KP-II when λ = 1 (weak surface tension).
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        lam: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, lam=lam, diff_mode=diff_mode)
        self.lam = lam

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]

        # First order space-time gradients
        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        g1 = g1_fn(points)
        u_x = g1[..., 0]
        u_y = g1[..., 1]
        u_t = g1[..., 2]

        # Iterative x-derivatives for the KdV core terms
        g2_x_fn = b.grad(lambda x: b.sum(g1_fn(x)[..., 0]), mode=self.diff_mode)
        u_xx = g2_x_fn(points)[..., 0]

        g3_x_fn = b.grad(lambda x: b.sum(g2_x_fn(x)[..., 0]), mode=self.diff_mode)
        u_xxx = g3_x_fn(points)[..., 0]

        # Second order y-derivative
        g2_y_fn = b.grad(lambda x: b.sum(g1_fn(x)[..., 1]), mode=self.diff_mode)
        u_yy = g2_y_fn(points)[..., 1]

        # Now compute outer d/dx derivatives for the mixed terms
        kdv_core_fn = lambda x: (
            g1_fn(x)[..., 2] + 6.0 * model_fn(x)[..., 0] * g1_fn(x)[..., 0] + g3_x_fn(x)[..., 0]
        )
        u_tx_plus_nonlinear = b.grad(lambda x: b.sum(kdv_core_fn(x)), mode=self.diff_mode)(points)[..., 0]

        return u_tx_plus_nonlinear + self.lam * u_yy


# ---------------------------------------------------------------------------
# 18. Porous Medium Equation (PME - 2D)
# ---------------------------------------------------------------------------

class PorousMediumResidual(PDEResidual):
    """
    ∂u/∂t - Δ(u^m) = 0   (m > 1)
    
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        m: float = 2.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, m=m, diff_mode=diff_mode)
        self.m = m

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        
        # Take laplacian of the non-linear physical state u^m
        u_m_fn = lambda x: model_fn(x)[..., 0] ** self.m
        lap_u_m = _laplacian(b, u_m_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        
        return u_t - lap_u_m


# ---------------------------------------------------------------------------
# 19. Biharmonic Equation (Steady-State 2D Elasticity)
# ---------------------------------------------------------------------------

class BiharmonicSteadyResidual(PDEResidual):
    """
    Δ²u = f(x, y)  -->  ∂⁴u/∂x⁴ + 2∂⁴u/∂x²∂y² + ∂⁴u/∂y⁴ = f(x, y)
    
    ``points`` columns: [x, y]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        source_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.source_fn = source_fn or (lambda xy: backend.zeros(xy.shape[:-1]))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        
        # Deconstruct biharmonic operator via nested spatial Laplacians
        lap_fn = lambda x: _laplacian(b, lambda y: model_fn(y)[..., 0], x, mode=self.diff_mode)
        biharmonic = _laplacian(b, lap_fn, points, mode=self.diff_mode)
        f = self.source_fn(points)
        
        return biharmonic - f


# ---------------------------------------------------------------------------
# 20. Boussinesq Wave Equation (Dispersive Long Waves)
# ---------------------------------------------------------------------------

class BoussinesqWaveResidual(PDEResidual):
    """
    ∂²u/∂t² - ∂²u/∂x² - 3 ∂²(u²)/∂x² - ∂⁴u/∂x⁴ = 0
    
    ``points`` columns: [x, t]
    """

    def __init__(self, backend: AbstractBackend, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, diff_mode=diff_mode)

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        # Time derivatives
        u_t_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_tt   = b.grad(lambda x: b.sum(u_t_fn(x)[..., 1]), mode=self.diff_mode)(points)[..., 1]

        # Pure spatial x-derivatives up to 4th order
        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_x = g1_fn(points)[..., 0]

        g2_fn = b.grad(lambda x: b.sum(g1_fn(x)[..., 0]), mode=self.diff_mode)
        u_xx = g2_fn(points)[..., 0]

        g3_fn = b.grad(lambda x: b.sum(g2_fn(x)[..., 0]), mode=self.diff_mode)
        g4_fn = b.grad(lambda x: b.sum(g3_fn(x)[..., 0]), mode=self.diff_mode)
        u_xxxx = g4_fn(points)[..., 0]

        # Spatial second derivative of the u^2 nonlinearity
        u_sq_fn = lambda x: model_fn(x)[..., 0] ** 2
        g1_sq_fn = b.grad(lambda x: b.sum(u_sq_fn(x)), mode=self.diff_mode)
        g2_sq_fn = b.grad(lambda x: b.sum(g1_sq_fn(x)[..., 0]), mode=self.diff_mode)
        u_sq_xx = g2_sq_fn(points)[..., 0]

        return u_tt - u_xx - 3.0 * u_sq_xx - u_xxxx


# ---------------------------------------------------------------------------
# 21. Euler-Tricomi Equation (Transonic Aerodynamics)
# ---------------------------------------------------------------------------

class EulerTricomiResidual(PDEResidual):
    """
    ∂²u/∂y² - x ∂²u/∂x² = 0
    
    Elliptic for x < 0, Hyperbolic for x > 0.
    ``points`` columns: [x, y]
    """

    def __init__(self, backend: AbstractBackend, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, diff_mode=diff_mode)

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        x = points[..., 0]

        g1_fn = b.grad(lambda pt: b.sum(model_fn(pt)), mode=self.diff_mode)
        
        u_xx = b.grad(lambda pt: b.sum(g1_fn(pt)[..., 0]), mode=self.diff_mode)(points)[..., 0]
        u_yy = b.grad(lambda pt: b.sum(g1_fn(pt)[..., 1]), mode=self.diff_mode)(points)[..., 1]

        return u_yy - x * u_xx


# ---------------------------------------------------------------------------
# 22. Fokker-Planck Equation (Stochastic Drift-Diffusion 2D)
# ---------------------------------------------------------------------------

class FokkerPlanck2DResidual(PDEResidual):
    """
    ∂p/∂t + ∇ · (μ p) - D Δp = 0
    
    Where probability density p(x,y,t) evolves under constant drift vector field μ.
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        drift: Sequence[float] = (0.1, 0.1),
        D: float = 0.05,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, drift=drift, D=D, diff_mode=diff_mode)
        self.drift = drift
        self.D = D

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        p = model_fn(points)[..., 0]
        
        p_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap_p = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        g1 = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)(points)
        p_x, p_y = g1[..., 0], g1[..., 1]

        # Divergence of drift field: div(drift * p) = mu_x * p_x + mu_y * p_y
        div_drift = self.drift[0] * p_x + self.drift[1] * p_y

        return p_t + div_drift - self.D * lap_p


# ---------------------------------------------------------------------------
# 23. Swift-Hohenberg Equation (Pattern Formation - 2D)
# ---------------------------------------------------------------------------

class SwiftHohenbergResidual(PDEResidual):
    """
    ∂u/∂t - r u + (1 + Δ)²u + u³ = 0
    
    Expanded: ∂u/∂t - r u + u + 2Δu + Δ²u + u³ = 0
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        r: float = 0.1,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, r=r, diff_mode=diff_mode)
        self.r = r

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]
        
        n_dims = points.shape[-1] - 1
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap_u = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=n_dims, mode=self.diff_mode)
        
        # Second order biharmonic component Δ²u
        lap_fn = lambda x: _laplacian(b, lambda y: model_fn(y)[..., 0], x, spatial_dims=n_dims, mode=self.diff_mode)
        lap2_u = _laplacian(b, lap_fn, points, spatial_dims=n_dims, mode=self.diff_mode)

        return u_t - self.r * u + u + 2.0 * lap_u + lap2_u + u**3


# ---------------------------------------------------------------------------
# 24. Black-Scholes Equation (2D Basket Option Pricing Model)
# ---------------------------------------------------------------------------

class BlackScholes2DResidual(PDEResidual):
    """
    ∂V/∂t + r S₁ ∂V/∂S₁ + r S₂ ∂V/∂S₂ + 0.5 σ₁² S₁² ∂²V/∂S₁² + 0.5 σ₂² S₂² ∂²V/∂S₂² + ρ σ₁ σ₂ S₁ S₂ ∂²V/∂S₁∂S₂ - r V = 0
    
    ``points`` columns: [S₁, S₂, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        r: float = 0.05,
        sigma1: float = 0.2,
        sigma2: float = 0.2,
        rho: float = 0.5,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, r=r, sigma1=sigma1, sigma2=sigma2, rho=rho, diff_mode=diff_mode)
        self.r = r
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        S1 = points[..., 0]
        S2 = points[..., 1]
        V = model_fn(points)[..., 0]

        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        g1 = g1_fn(points)
        V_S1, V_S2, V_t = g1[..., 0], g1[..., 1], g1[..., 2]

        V_S1_fn = lambda x: g1_fn(x)[..., 0]
        V_S2_fn = lambda x: g1_fn(x)[..., 1]

        V_S1S1 = b.grad(lambda x: b.sum(V_S1_fn(x)), mode=self.diff_mode)(points)[..., 0]
        V_S2S2 = b.grad(lambda x: b.sum(V_S2_fn(x)), mode=self.diff_mode)(points)[..., 1]
        V_S1S2 = b.grad(lambda x: b.sum(V_S1_fn(x)), mode=self.diff_mode)(points)[..., 1]

        drift = self.r * S1 * V_S1 + self.r * S2 * V_S2
        diffusion = 0.5 * (self.sigma1**2) * (S1**2) * V_S1S1 + 0.5 * (self.sigma2**2) * (S2**2) * V_S2S2
        cross_term = self.rho * self.sigma1 * self.sigma2 * S1 * S2 * V_S1S2

        return V_t + drift + diffusion + cross_term - self.r * V


# ---------------------------------------------------------------------------
# 25. Regularised Long Wave (RLW) Equation
# ---------------------------------------------------------------------------

class RegularisedLongWaveResidual(PDEResidual):
    """
    ∂u/∂t + ∂u/∂x + u ∂u/∂x - μ ∂³u/∂x²∂t = 0
    
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mu: float = 0.1,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, mu=mu, diff_mode=diff_mode)
        self.mu = mu

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u = model_fn(points)[..., 0]

        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        g1 = g1_fn(points)
        u_x = g1[..., 0]
        u_t = g1[..., 1]

        # Iterative spatial chain to compute the mixed derivative (∂³u / ∂x²∂t)
        u_x_fn = lambda x: g1_fn(x)[..., 0]
        u_xx_fn = lambda x: b.grad(lambda y: b.sum(u_x_fn(y)), mode=self.diff_mode)(x)[..., 0]
        u_xxt = b.grad(lambda x: b.sum(u_xx_fn(x)), mode=self.diff_mode)(points)[..., 1]

        return u_t + u_x + u * u_x - self.mu * u_xxt


# ---------------------------------------------------------------------------
# 26. Gierer-Meinhardt Activator-Inhibitor System (2D Structure)
# ---------------------------------------------------------------------------

class GiererMeinhardtResidual(PDEResidual):
    """
    ∂a/∂t = D_a Δa + a² / i - μ_a a + ρ_a
    ∂i/∂t = D_i Δi + a² - μ_i i
    
    ``model_fn(xyt)`` → [a, i]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        D_a: float = 0.01,
        D_i: float = 0.4,
        mu_a: float = 0.01,
        mu_i: float = 0.02,
        rho_a: float = 0.001,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, D_a=D_a, D_i=D_i, mu_a=mu_a, mu_i=mu_i, rho_a=rho_a, diff_mode=diff_mode)
        self.D_a, self.D_i = D_a, D_i
        self.mu_a, self.mu_i = mu_a, mu_i
        self.rho_a = rho_a

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        a_fn = lambda x: model_fn(x)[..., 0:1]
        i_fn = lambda x: model_fn(x)[..., 1:2]

        a_t = _time_deriv(b, lambda x: a_fn(x)[..., 0], points, mode=self.diff_mode)
        i_t = _time_deriv(b, lambda x: i_fn(x)[..., 0], points, mode=self.diff_mode)

        lap_a = _laplacian(b, lambda x: a_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_i = _laplacian(b, lambda x: i_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        a = model_fn(points)[..., 0]
        i = model_fn(points)[..., 1]

        res_a = a_t - self.D_a * lap_a - (a**2 / (i + 1e-6) - self.mu_a * a + self.rho_a)
        res_i = i_t - self.D_i * lap_i - (a**2 - self.mu_i * i)

        return b.stack([res_a, res_i], axis=-1)


# ---------------------------------------------------------------------------
# 27. Gray-Scott Autocatalytic System (Reaction-Diffusion 2D)
# ---------------------------------------------------------------------------

class GrayScottResidual(PDEResidual):
    """
    ∂u/∂t = D_u Δu - u v² + F (1 - u)
    ∂v/∂t = D_v Δv + u v² - (F + k) v
    
    ``model_fn(xyt)`` → [u, v]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        D_u: float = 0.16,
        D_v: float = 0.08,
        F: float = 0.035,
        k: float = 0.060,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, D_u=D_u, D_v=D_v, F=F, k=k, diff_mode=diff_mode)
        self.D_u, self.D_v = D_u, D_v
        self.F, self.k = F, k

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        u = model_fn(points)[..., 0]
        v = model_fn(points)[..., 1]
        uv_sq = u * (v**2)

        res_u = u_t - self.D_u * lap_u + uv_sq - self.F * (1.0 - u)
        res_v = v_t - self.D_v * lap_v - uv_sq + (self.F + self.k) * v

        return b.stack([res_u, res_v], axis=-1)


# ---------------------------------------------------------------------------
# 28. Viscous Wave Equation (Damped Acoustic Waves)
# ---------------------------------------------------------------------------

class ViscousWaveResidual(PDEResidual):
    """
    ∂²u/∂t² - c² Δu - ν Δ(∂u/∂t) = 0
    
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        c: float = 1.0,
        nu: float = 0.05,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, c=c, nu=nu, diff_mode=diff_mode)
        self.c = c
        self.nu = nu

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        u_t_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        u_t_wrapped = lambda x: u_t_fn(x)[..., -1]
        u_tt = b.grad(lambda x: b.sum(u_t_wrapped(x)), mode=self.diff_mode)(points)[..., -1]

        lap_u = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_ut = _laplacian(b, u_t_wrapped, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        return u_tt - (self.c**2) * lap_u - self.nu * lap_ut


# ---------------------------------------------------------------------------
# 29. Radhakrishnan-Kundu-Lakshmanan (RKL) Equation (Optical Fibres)
# ---------------------------------------------------------------------------

class RadhakrishnanKunduLakshmananResidual(PDEResidual):
    """
    i ∂ψ/∂t + α ∂²ψ/∂x² + β |ψ|² ψ + i γ ∂³ψ/∂x³ = 0
    
    Split profile into real components where ψ = u + iv:
        ∂u/∂t = -α ∂²v/∂x² - β (u² + v²) v - γ ∂³u/∂x³
        ∂v/∂t =  α ∂²u/∂x² + β (u² + v²) u - γ ∂³v/∂x³
        
    ``model_fn(xt)`` → [u, v]
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.1,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, alpha=alpha, beta=beta, gamma=gamma, diff_mode=diff_mode)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)

        # 2nd and 3rd order spatials for Real Component u
        g1_u_fn = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)
        g2_u_fn = b.grad(lambda x: b.sum(g1_u_fn(x)[..., 0]), mode=self.diff_mode)
        u_xx = g2_u_fn(points)[..., 0]
        u_xxx = b.grad(lambda x: b.sum(g2_u_fn(x)[..., 0]), mode=self.diff_mode)(points)[..., 0]

        # 2nd and 3rd order spatials for Imaginary Component v
        g1_v_fn = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)
        g2_v_fn = b.grad(lambda x: b.sum(g1_v_fn(x)[..., 0]), mode=self.diff_mode)
        v_xx = g2_v_fn(points)[..., 0]
        v_xxx = b.grad(lambda x: b.sum(g2_v_fn(x)[..., 0]), mode=self.diff_mode)(points)[..., 0]

        u = model_fn(points)[..., 0]
        v = model_fn(points)[..., 1]
        mag_sq = u**2 + v**2

        # res_u is the equation governing du/dt (real part); res_v governs dv/dt (imag part)
        res_u = u_t + self.alpha * v_xx + self.beta * mag_sq * v + self.gamma * u_xxx
        res_v = v_t - self.alpha * u_xx - self.beta * mag_sq * u + self.gamma * v_xxx

        return b.stack([res_u, res_v], axis=-1)


# ---------------------------------------------------------------------------
# 30. Complex Ginzburg-Landau Equation (CGLE - 2D)
# ---------------------------------------------------------------------------

class ComplexGinzburgLandau2DResidual(PDEResidual):
    """
    ∂ψ/∂t = ψ + (1 + i c₁) Δψ - (1 + i c₃) |ψ|² ψ
    
    Split into real components where ψ = u + iv:
        ∂u/∂t = u - c₁ Δv + Δu - (u² + v²) u + c₃ (u² + v²) v
        ∂v/∂t = v + c₁ Δu + Δv - (u² + v²) v - c₃ (u² + v²) u
        
    ``model_fn(xyt)`` → [u, v]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        c1: float = 0.5,
        c3: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, c1=c1, c3=c3, diff_mode=diff_mode)
        self.c1 = c1
        self.c3 = c3

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        u = model_fn(points)[..., 0]
        v = model_fn(points)[..., 1]
        mag_sq = u**2 + v**2

        res_real = u_t - (u + lap_u - self.c1 * lap_v - mag_sq * u + self.c3 * mag_sq * v)
        res_imag = v_t - (v + lap_v + self.c1 * lap_u - mag_sq * v - self.c3 * mag_sq * u)

        return b.stack([res_real, res_imag], axis=-1)

# ---------------------------------------------------------------------------
# 31. Drift-Diffusion Poisson System (Semiconductor Transport / Physics)
# ---------------------------------------------------------------------------

class DriftDiffusionPoissonResidual(PDEResidual):
    """
    Simulates charge carrier transport in semiconductors (Solar cells, Transistors).
    Equations:
        ∂n/∂t = ∇ · (D_n ∇n - μ_n n ∇ψ)
        -Δψ = q/ε (n - p - C_d)

    ``model_fn(xt)`` → [n, ψ] (Electron density, Electrostatic potential)
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        D_n: float = 1.0,
        mu_n: float = 1.0,
        scaling_factor: float = 1.0,  # q/ε
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, D_n=D_n, mu_n=mu_n, scaling_factor=scaling_factor, diff_mode=diff_mode)
        self.D_n = D_n
        self.mu_n = mu_n
        self.scale = scaling_factor

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        
        n_fn = lambda x: model_fn(x)[..., 0:1]
        psi_fn = lambda x: model_fn(x)[..., 1:2]

        n_t = _time_deriv(b, lambda x: n_fn(x)[..., 0], points, mode=self.diff_mode)
        lap_n = _laplacian(b, lambda x: n_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_psi = _laplacian(b, lambda x: psi_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        # Gradients for drift term: div(n * grad(ψ)) = grad(n)·grad(ψ) + n * lap(ψ)
        g_n = b.grad(lambda x: b.sum(n_fn(x)), mode=self.diff_mode)(points)[..., 0]
        g_psi = b.grad(lambda x: b.sum(psi_fn(x)), mode=self.diff_mode)(points)[..., 0]
        
        drift_div = g_n * g_psi + n_fn(points)[..., 0] * lap_psi
        
        res_electron = n_t - (self.D_n * lap_n - self.mu_n * drift_div)
        res_poisson = -lap_psi - self.scale * n_fn(points)[..., 0]  # Simplified intrinsic case (p=0, C_d=0)

        return b.stack([res_electron, res_poisson], axis=-1)


# ---------------------------------------------------------------------------
# 32. 2D Shallow Water Equations (Climate Modeling / Oceanography)
# ---------------------------------------------------------------------------

class ShallowWater2DResidual(PDEResidual):
    """
    Used for tsunami tracking, atmospheric flows, and climate simulation.
    Equations:
        ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
        ∂(hu)/∂t + ∂(hu² + 0.5gh²)/∂x + ∂(huv)/∂y = 0
        ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + 0.5gh²)/∂y = 0

    ``model_fn(xyt)`` → [h, u, v] (Water height, x-velocity, y-velocity)
    ``points`` columns: [x, y, t]
    """

    def __init__(self, backend: AbstractBackend, g: float = 9.81, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, g=g, diff_mode=diff_mode)
        self.g = g

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        
        h_fn = lambda x: model_fn(x)[..., 0:1]
        u_fn = lambda x: model_fn(x)[..., 1:2]
        v_fn = lambda x: model_fn(x)[..., 2:3]

        h, u, v = model_fn(points)[..., 0], model_fn(points)[..., 1], model_fn(points)[..., 2]

        g_h = b.grad(lambda x: b.sum(h_fn(x)), mode=self.diff_mode)(points)
        g_u = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)
        g_v = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)

        h_x, h_y, h_t = g_h[..., 0], g_h[..., 1], g_h[..., 2]
        u_x, u_y, u_t = g_u[..., 0], g_u[..., 1], g_u[..., 2]
        v_x, v_y, v_t = g_v[..., 0], g_v[..., 1], g_v[..., 2]

        # Continuity
        res_mass = h_t + h_x * u + h * u_x + h_y * v + h * v_y
        
        # Momentum X & Y
        res_mom_x = u_t + u * u_x + v * u_y + self.g * h_x
        res_mom_y = v_t + u * v_x + v * v_y + self.g * h_y

        return b.stack([res_mass, res_mom_x, res_mom_y], axis=-1)


# ---------------------------------------------------------------------------
# 33. Heston Volatility PDE (Quantitative Finance / Option Pricing)
# ---------------------------------------------------------------------------

class HestonVolatilityResidual(PDEResidual):
    """
    Standard option pricing model mapping asset price (S) and its variance (v).
    ``points`` columns: [S, v, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        r: float = 0.03,      # Risk-free rate
        kappa: float = 2.0,  # Mean reversion speed
        theta: float = 0.04,  # Long-term variance
        sigma: float = 0.3,   # Volatility of volatility
        rho: float = -0.7,  # Asset-volatility correlation
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho, diff_mode=diff_mode)
        self.r, self.kappa, self.theta, self.sigma, self.rho = r, kappa, theta, sigma, rho

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        S, v = points[..., 0], points[..., 1]
        U = model_fn(points)[..., 0]

        g1_fn = b.grad(lambda x: b.sum(model_fn(x)), mode=self.diff_mode)
        g1 = g1_fn(points)
        U_S, U_v, U_t = g1[..., 0], g1[..., 1], g1[..., 2]

        U_S_fn = lambda x: g1_fn(x)[..., 0]
        U_v_fn = lambda x: g1_fn(x)[..., 1]

        U_SS = b.grad(lambda x: b.sum(U_S_fn(x)), mode=self.diff_mode)(points)[..., 0]
        U_vv = b.grad(lambda x: b.sum(U_v_fn(x)), mode=self.diff_mode)(points)[..., 1]
        U_Sv = b.grad(lambda x: b.sum(U_S_fn(x)), mode=self.diff_mode)(points)[..., 1]

        drift = self.r * S * U_S + self.kappa * (self.theta - v) * U_v
        diffusion = 0.5 * v * (S**2) * U_SS + 0.5 * (self.sigma**2) * v * U_vv
        cross = self.rho * self.sigma * v * S * U_Sv

        return U_t + drift + diffusion + cross - self.r * U


# ---------------------------------------------------------------------------
# 34. Phase-Field Crystal Equation (Material Science / Metallurgy)
# ---------------------------------------------------------------------------

class PhaseFieldCrystalResidual(PDEResidual):
    """
    Models crystal atomic density configurations over diffusive timelines.
    ∂ϕ/∂t - Δ( ϕ³ - ϵϕ + 2Δϕ + Δ²ϕ ) = 0
    
    ``points`` columns: [x, y, t]
    """

    def __init__(self, backend: AbstractBackend, epsilon: float = 0.25, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, epsilon=epsilon, diff_mode=diff_mode)
        self.epsilon = epsilon

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        phi = model_fn(points)[..., 0]
        
        n_dims = points.shape[-1] - 1
        phi_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)
        lap_phi = _laplacian(b, lambda x: model_fn(x)[..., 0], points, spatial_dims=n_dims, mode=self.diff_mode)
        
        lap_fn = lambda x: _laplacian(b, lambda y: model_fn(y)[..., 0], x, spatial_dims=n_dims, mode=self.diff_mode)
        lap2_phi = _laplacian(b, lap_fn, points, spatial_dims=n_dims, mode=self.diff_mode)

        # Internal chemical potential evaluation wrapper
        # mu = phi^3 - epsilon*phi + 2*lap(phi) + lap2(phi)
        mu_fn = lambda x: (
            model_fn(x)[..., 0]**3 - self.epsilon * model_fn(x)[..., 0]
            + 2.0 * _laplacian(b, lambda y: model_fn(y)[..., 0], x, spatial_dims=n_dims, mode=self.diff_mode)
            + _laplacian(
                b,
                lambda y: _laplacian(b, lambda z: model_fn(z)[..., 0], y, spatial_dims=n_dims, mode=self.diff_mode),
                x,
                spatial_dims=n_dims,
                mode=self.diff_mode,
            )
        )
        
        lap_mu = _laplacian(b, mu_fn, points, spatial_dims=n_dims, mode=self.diff_mode)
        return phi_t - lap_mu


# ---------------------------------------------------------------------------
# 35. 2D Brinkman-Extended Darcy Flow (Porous Matrix Media Aerodynamics)
# ---------------------------------------------------------------------------

class BrinkmanDarcyFlowResidual(PDEResidual):
    """
    Sub-surface hydrology and oil reservoir flow tracking matrix.
    -μ_eff Δu + μ/K u + ∇p = f
    
    ``model_fn(xy)`` → [u, v, p]
    ``points`` columns: [x, y]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mu_eff: float = 0.05,
        darcy_coeff: float = 10.0,  # μ/K
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, mu_eff=mu_eff, darcy_coeff=darcy_coeff, diff_mode=diff_mode)
        self.mu_eff = mu_eff
        self.darcy = darcy_coeff

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]
        p_fn = lambda x: model_fn(x)[..., 2:3]

        g_p = b.grad(lambda x: b.sum(p_fn(x)), mode=self.diff_mode)(points)
        p_x, p_y = g_p[..., 0], g_p[..., 1]

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)

        u, v = model_fn(points)[..., 0], model_fn(points)[..., 1]
        g_u = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)[..., 0]
        g_v = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)[..., 1]

        res_u = -self.mu_eff * lap_u + self.darcy * u + p_x
        res_v = -self.mu_eff * lap_v + self.darcy * v + p_y
        res_div = g_u + g_v

        return b.stack([res_u, res_v, res_div], axis=-1)

# ---------------------------------------------------------------------------
# 36. Anisotropic Perona-Malik PDE (Computer Vision / Image Denoising)
# ---------------------------------------------------------------------------

class PeronaMalikResidual(PDEResidual):
    """
    Edge-preserving diffusion filter used in digital imaging.
    ∂u/∂t - ∇ · ( c(|∇u|²) ∇u ) = 0  where c(x) = exp(-x/K²)
    
    ``points`` columns: [x, y, t]
    """

    def __init__(self, backend: AbstractBackend, K: float = 0.1, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, K=K, diff_mode=diff_mode)
        self.K_sq = K ** 2

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_t = _time_deriv(b, lambda x: model_fn(x)[..., 0:1], points, mode=self.diff_mode)

        # Helper to compute components of the flux vector: flux = c(|∇u|²) * ∇u
        def flux_components(pt: Tensor) -> tuple[Tensor, Tensor]:
            g_u = b.grad(lambda x: b.sum(model_fn(x)[..., 0]), mode=self.diff_mode)(pt)
            u_x, u_y = g_u[..., 0], g_u[..., 1]
            grad_mag_sq = u_x**2 + u_y**2
            c_coef = b.exp(-grad_mag_sq / self.K_sq)
            return c_coef * u_x, c_coef * u_y

        # Wrap component functions for auto-diff tracking
        flux_x_fn = lambda pt: flux_components(pt)[0]
        flux_y_fn = lambda pt: flux_components(pt)[1]

        # Compute divergence: ∂(flux_x)/∂x + ∂(flux_y)/∂y
        div_x = b.grad(lambda x: b.sum(flux_x_fn(x)), mode=self.diff_mode)(points)[..., 0]
        div_y = b.grad(lambda x: b.sum(flux_y_fn(x)), mode=self.diff_mode)(points)[..., 1]

        return u_t - (div_x + div_y)


# ---------------------------------------------------------------------------
# 37. Viscous Burgers 2D Vector System (Turbulence Modeling Core)
# ---------------------------------------------------------------------------

class Burgers2DResidual(PDEResidual):
    """
    Simplified vehicle for supersonic flow tracking and shock profile validation.
    ∂u/∂t + u ∂u/∂x + v ∂u/∂y - ν Δu = 0
    ∂v/∂t + u ∂v/∂x + v ∂v/∂y - ν Δv = 0
    
    ``model_fn(xyt)`` → [u, v]
    ``points`` columns: [x, y, t]
    """

    def __init__(self, backend: AbstractBackend, nu: float = 0.01, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, nu=nu, diff_mode=diff_mode)
        self.nu = nu

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]

        u, v = model_fn(points)[..., 0], model_fn(points)[..., 1]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)

        g_u = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)
        g_v = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)
        u_x, u_y = g_u[..., 0], g_u[..., 1]
        v_x, v_y = g_v[..., 0], g_v[..., 1]

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        res_u = u_t + u * u_x + v * u_y - self.nu * lap_u
        res_v = v_t + u * v_x + v * v_y - self.nu * lap_v

        return b.stack([res_u, res_v], axis=-1)


# ---------------------------------------------------------------------------
# 38. Boussinesq Approximation for Thermal Convection (Mantle / Weather)
# ---------------------------------------------------------------------------

class BoussinesqConvectionResidual(PDEResidual):
    """
    Couples Navier-Stokes velocity vectors with fluid heat transport.
    Active buoyancy engine for Earth's mantle or industrial furnace engineering.
    
    ``model_fn(xyt)`` → [u, v, p, T]
    ``points`` columns: [x, y, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        Pr: float = 0.71,   # Prandtl number
        Ra: float = 1000.0,  # Rayleigh number
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, Pr=Pr, Ra=Ra, diff_mode=diff_mode)
        self.Pr = Pr
        self.Ra = Ra

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u, v = model_fn(points)[..., 0], model_fn(points)[..., 1]
        T = model_fn(points)[..., 3]

        u_fn = lambda x: model_fn(x)[..., 0:1]
        v_fn = lambda x: model_fn(x)[..., 1:2]
        p_fn = lambda x: model_fn(x)[..., 2:3]
        T_fn = lambda x: model_fn(x)[..., 3:4]

        u_t = _time_deriv(b, lambda x: u_fn(x)[..., 0], points, mode=self.diff_mode)
        v_t = _time_deriv(b, lambda x: v_fn(x)[..., 0], points, mode=self.diff_mode)
        T_t = _time_deriv(b, lambda x: T_fn(x)[..., 0], points, mode=self.diff_mode)

        g_u = b.grad(lambda x: b.sum(u_fn(x)), mode=self.diff_mode)(points)
        g_v = b.grad(lambda x: b.sum(v_fn(x)), mode=self.diff_mode)(points)
        g_p = b.grad(lambda x: b.sum(p_fn(x)), mode=self.diff_mode)(points)
        g_T = b.grad(lambda x: b.sum(T_fn(x)), mode=self.diff_mode)(points)

        u_x, u_y = g_u[..., 0], g_u[..., 1]
        v_x, v_y = g_v[..., 0], g_v[..., 1]
        p_x, p_y = g_p[..., 0], g_p[..., 1]
        T_x, T_y = g_T[..., 0], g_T[..., 1]

        lap_u = _laplacian(b, lambda x: u_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_v = _laplacian(b, lambda x: v_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_T = _laplacian(b, lambda x: T_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        res_u = u_t + u * u_x + v * u_y + p_x - self.Pr * lap_u
        res_v = v_t + u * v_x + v * v_y + p_y - self.Pr * lap_v - self.Ra * self.Pr * T
        res_div = u_x + v_y
        res_T = T_t + u * T_x + v * T_y - lap_T

        return b.stack([res_u, res_v, res_div, res_T], axis=-1)


# ---------------------------------------------------------------------------
# 39. Phase-Field Allen-Cahn Dendritic Solidification (3D Metallurgy)
# ---------------------------------------------------------------------------

class DendriticSolidificationResidual(PDEResidual):
    """
    Simulates crystalline structures during casting or additive manufacturing.
    
    ``model_fn(xyzt)`` → [phi, T] (Phase identity field, Temperature field)
    ``points`` columns: [x, y, z, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        epsilon: float = 0.01,
        K: float = 1.0,  # Latent heat absorption coefficient
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, epsilon=epsilon, K=K, diff_mode=diff_mode)
        self.eps_sq = epsilon ** 2
        self.K = K

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        phi = model_fn(points)[..., 0]
        T = model_fn(points)[..., 1]

        phi_fn = lambda x: model_fn(x)[..., 0:1]
        T_fn = lambda x: model_fn(x)[..., 1:2]

        phi_t = _time_deriv(b, lambda x: phi_fn(x)[..., 0], points, mode=self.diff_mode)
        T_t = _time_deriv(b, lambda x: T_fn(x)[..., 0], points, mode=self.diff_mode)

        lap_phi = _laplacian(b, lambda x: phi_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)
        lap_T = _laplacian(b, lambda x: T_fn(x)[..., 0], points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        # Well potential function for phase boundary sorting: m(T) = arctan(K * T)
        m_T = b.atan(self.K * T)
        res_phi = phi_t - self.eps_sq * lap_phi - phi * (1.0 - phi) * (phi - 0.5 + m_T)
        res_T = T_t - lap_T - self.K * phi_t

        return b.stack([res_phi, res_T], axis=-1)


# ---------------------------------------------------------------------------
# 40. Relativistic Hydrodynamics (Euler-Einstein Baseline Field Core)
# ---------------------------------------------------------------------------

class RelativisticFluidCoreResidual(PDEResidual):
    """
    Models plasma acceleration fields and high-energy astrophysic flows.
    Equation: ∂/∂t (γ² h - p) + ∂/∂x (γ² h u) = 0
    
    ``model_fn(xt)`` → [h, u] (Enthalpy density profile, Relative velocity)
    ``points`` columns: [x, t]
    """

    def __init__(self, backend: AbstractBackend, speed_of_light: float = 1.0, *, diff_mode: str = "reverse") -> None:
        super().__init__(backend, c=speed_of_light, diff_mode=diff_mode)
        self.c = speed_of_light
        self.c_sq = speed_of_light ** 2

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        # The network's raw second output channel is an unconstrained real
        # number, but the physical relative velocity u must satisfy |u| < c
        # for the Lorentz factor to be defined at all. At random init the
        # raw channel routinely exceeds c, which drove 1 - u^2/c^2 negative
        # and produced NaN through sqrt(negative) - not a training-stability
        # issue but a domain violation of the model's own output.
        #
        # u = c * tanh(raw) is not an approximation of the physics: tanh
        # is a bijection from the raw unconstrained network output onto the
        # exact physically-admissible velocity range (-c, c), so u is always
        # in-domain by construction and 1 - u^2/c^2 is always strictly in
        # (0, 1]. No epsilon fudge is needed, and none of the conservation
        # laws below are touched - u is still exactly "the relative
        # velocity field the model predicts", just parameterised so it can
        # never leave the physically valid range.
        def relative_velocity(pt: Tensor) -> Tensor:
            raw_u = model_fn(pt)[..., 1]
            return self.c * b.tanh(raw_u)

        # Wrapped calculations to maintain pure functions for auto-diff engine tracking
        def state_variables(pt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            h_val = model_fn(pt)[..., 0]
            u_val = relative_velocity(pt)
            lorentz = 1.0 / b.sqrt(1.0 - (u_val**2 / self.c_sq))
            p_val = 0.333 * h_val  # Ultra-relativistic equation of state ideal gas approximation
            return h_val, u_val, lorentz * h_val - p_val

        # NOTE: this is the relativistic energy density D = gamma^2*h - p,
        # not a mass density; renamed from mass_density_fn for clarity.
        energy_density_fn = lambda pt: state_variables(pt)[2]

        def momentum_density_fn(pt: Tensor) -> Tensor:
            h_val = model_fn(pt)[..., 0]
            u_val = relative_velocity(pt)
            lorentz_sq = 1.0 / (1.0 - (u_val**2 / self.c_sq))
            return lorentz_sq * h_val * u_val

        def momentum_flux_fn(pt: Tensor) -> Tensor:
            # Momentum flux = gamma^2 * h * u^2 + p
            h_val = model_fn(pt)[..., 0]
            u_val = relative_velocity(pt)
            lorentz_sq = 1.0 / (1.0 - (u_val**2 / self.c_sq))
            p_val = 0.333 * h_val
            return lorentz_sq * h_val * u_val**2 + p_val

        # Energy-continuity equation: dD/dt + d(momentum_density)/dx = 0
        energy_t = _time_deriv(b, energy_density_fn, points, mode=self.diff_mode)
        momentum_flux_x = b.grad(lambda x: b.sum(momentum_density_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_energy = energy_t + momentum_flux_x

        # Momentum equation: d(momentum_density)/dt + d(momentum_flux)/dx = 0
        # Model outputs [h, u] (2 fields), so both the energy and momentum
        # residual equations are returned to fully constrain the system.
        momentum_t = _time_deriv(b, momentum_density_fn, points, mode=self.diff_mode)
        stress_x = b.grad(lambda x: b.sum(momentum_flux_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_momentum = momentum_t + stress_x

        return b.stack([res_energy, res_momentum], axis=-1)


# ---------------------------------------------------------------------------
# 41. Einstein Field Equations (full nonlinear, general 4-D metric)
# ---------------------------------------------------------------------------
#
# This block is fully self-contained. It introduces its own tensor-calculus
# helpers (``_batched_jacobian``, ``_assemble_symmetric4``, ``_det_n``,
# ``_cofactor_matrix``, ``_inverse4``) rather than reusing or modifying the
# scalar-field helpers above (``_grad``, ``_grad2``, ``_laplacian``,
# ``_time_deriv``, ``_spatial_grad``), which every existing residual class
# depends on and which are left completely untouched.

def _batched_jacobian(
    backend: AbstractBackend,
    vec_fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    out_dim: int,
    *,
    mode: str = "reverse",
) -> Tensor:
    """
    Batched Jacobian of a vector-valued ``vec_fn(x) -> (..., out_dim)``
    w.r.t. ``inputs`` (..., in_dim). Returns (..., out_dim, in_dim).

    New, GR-specific helper. Uses the same "sum-trick per output slice"
    idiom already used inline for coupled systems elsewhere in this file
    (e.g. NavierStokesResidual's per-component grads), just generalised to
    an arbitrary number of output components and factored out so the
    Einstein-tensor machinery below can call it twice (once for ∂g, once
    for ∂Γ) instead of duplicating the loop.
    """
    rows = []
    for c in range(out_dim):
        def _fc(x: Tensor, _c: int = c) -> Tensor:
            return backend.sum(vec_fn(x)[..., _c])
        rows.append(backend.grad(_fc, mode=mode)(inputs))  # (..., in_dim)
    return backend.stack(rows, axis=-2)  # (..., out_dim, in_dim)


# Symmetric 4x4 <-> 10-component packing, shared by the metric, its
# derivatives, and the residual output. Order: tt, tx, ty, tz, xx, xy, xz,
# yy, yz, zz (i.e. row-major over the upper triangle, t=0,x=1,y=2,z=3).
_G_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3),
    (3, 3),
)
_G_PAIR_INDEX: Dict[Tuple[int, int], int] = {}
for _k, (_i, _j) in enumerate(_G_PAIRS):
    _G_PAIR_INDEX[(_i, _j)] = _k
    _G_PAIR_INDEX[(_j, _i)] = _k


def _det_n(M: list) -> Tensor:
    """Determinant of a small (<=4) nested-list matrix of batched scalar
    tensors, via recursive Laplace expansion. Pure elementwise arithmetic
    only, so it is differentiable through whatever backend produced the
    entries of ``M``."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    total = None
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        term = M[0][j] * _det_n(minor)
        if j % 2 == 1:
            term = -term
        total = term if total is None else total + term
    return total


def _cofactor_matrix(M: list) -> list:
    n = len(M)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [row[:j] + row[j + 1:] for r, row in enumerate(M) if r != i]
            cof = _det_n(minor)
            C[i][j] = cof if (i + j) % 2 == 0 else -cof
    return C


def _inverse4(M: list, eps: float = 1e-9) -> Tuple[list, Tensor]:
    """Closed-form 4x4 inverse via adjugate/determinant, operating on a
    nested-list matrix of batched scalar tensors. Returns (inverse, det).

    ``eps`` nudges the determinant away from exactly zero purely as a
    numerical-stability guard against early-training metric degeneracy
    (e.g. a randomly initialised network momentarily predicting a
    near-singular g_munu); it does not alter the physics once training
    has converged toward a genuine Lorentzian metric.
    """
    det = _det_n(M)
    det_safe = det + eps
    cof = _cofactor_matrix(M)
    inv = [[cof[j][i] / det_safe for j in range(4)] for i in range(4)]
    return inv, det


class EinsteinFieldResidual(PDEResidual):
    r"""
    Full nonlinear Einstein Field Equations, general 4-D metric (no symmetry
    assumption — vacuum, matter-coupled, or cosmological-constant cases are
    all supported through ``stress_energy_fn`` / ``cosmological_constant``):

        G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi T_{\mu\nu}

    where G_{\mu\nu} = R_{\mu\nu} - (1/2) g_{\mu\nu} R is built from the
    Christoffel symbols, Riemann tensor and Ricci tensor/scalar of the
    metric predicted by the network, all obtained via nested automatic
    differentiation (no finite differences, no hand-derived Christoffels
    for a specific ansatz — this works for an arbitrary metric).

    ``model_fn(x)`` -> 10 outputs: the independent components of the
    symmetric metric g_{\mu\nu}, ordered [g_tt, g_tx, g_ty, g_tz, g_xx,
    g_xy, g_xz, g_yy, g_yz, g_zz].
    ``points`` columns: [t, x, y, z]  (geometric units, c = G = 1 by
    default; rescale ``points``/outputs if you need SI units).

    Returns the 10 independent residual components, same stacking
    convention as the other coupled-system residuals in this file
    (e.g. ``NavierStokesResidual``).

    Notes on cost and initialisation
    ---------------------------------
    This is the heaviest residual in the file by design: Riemann is a
    second-derivative object, so evaluating it needs two nested levels of
    automatic differentiation over a 10-component (metric) then
    64-component (Christoffel) field. Expect O(few hundred) backend.grad
    calls per residual evaluation. Prefer small collocation batches and/or
    ``diff_mode="forward"`` or ``"taylor"`` for this residual specifically.

    Also, because ``g_{\mu\nu}`` must stay non-degenerate (invertible) and
    Lorentzian-signature to mean anything physically, a raw untrained MLP
    will typically produce a near-singular or wrong-signature metric at
    initialisation. In practice you'll want the model's output layer
    biased toward the Minkowski metric diag(-1, 1, 1, 1) (e.g. via
    ``model_fn``'s output bias, or a residual/skip parameterisation
    g = eta + h(x) with h initialised near zero) rather than feeding this
    residual's raw output straight from an unbiased network.
    """

    def __init__(
        self,
        backend: AbstractBackend,
        cosmological_constant: float = 0.0,
        stress_energy_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(
            backend,
            cosmological_constant=cosmological_constant,
            diff_mode=diff_mode,
        )
        self.Lambda = cosmological_constant
        # stress_energy_fn(points) -> (..., 10) T_{mu nu} components, same
        # packing as the metric. Defaults to vacuum (T = 0).
        self.stress_energy_fn = stress_energy_fn

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        mode = self.diff_mode

        def g_components_fn(x: Tensor) -> Tensor:
            return model_fn(x)[..., :10]

        def christoffel_flat_fn(x: Tensor) -> Tensor:
            g10 = g_components_fn(x)                                   # (..., 10)
            dg = _batched_jacobian(b, g_components_fn, x, 10, mode=mode)  # (..., 10, 4)

            g_list = [g10[..., k] for k in range(10)]

            def d_g(a: int, c: int, deriv: int) -> Tensor:
                return dg[..., _G_PAIR_INDEX[(a, c)], deriv]

            M = [[g_list[_G_PAIR_INDEX[(i, j)]] for j in range(4)] for i in range(4)]
            ginv, _det = _inverse4(M)

            gamma_rows = []  # flattened (lambda, mu, nu), row-major, 64 entries
            for lam in range(4):
                for mu in range(4):
                    for nu in range(4):
                        s = None
                        for sig in range(4):
                            term = ginv[lam][sig] * (
                                d_g(sig, nu, mu) + d_g(sig, mu, nu) - d_g(mu, nu, sig)
                            )
                            s = term if s is None else s + term
                        gamma_rows.append(0.5 * s)
            return b.stack(gamma_rows, axis=-1)  # (..., 64)

        # Metric + first derivatives at the collocation points.
        g10 = g_components_fn(points)
        dg = _batched_jacobian(b, g_components_fn, points, 10, mode=mode)
        g_list = [g10[..., k] for k in range(10)]

        def d_g(a: int, c: int, deriv: int) -> Tensor:
            return dg[..., _G_PAIR_INDEX[(a, c)], deriv]

        M = [[g_list[_G_PAIR_INDEX[(i, j)]] for j in range(4)] for i in range(4)]
        ginv, _det = _inverse4(M)

        # Christoffel symbols (value) and their derivatives, both via
        # autodiff over the same analytic formula — no separately
        # hand-coded second-derivative-of-g bookkeeping.
        gamma_flat = christoffel_flat_fn(points)                              # (..., 64)
        dgamma = _batched_jacobian(b, christoffel_flat_fn, points, 64, mode=mode)  # (..., 64, 4)

        def gamma(lam: int, mu: int, nu: int) -> Tensor:
            return gamma_flat[..., lam * 16 + mu * 4 + nu]

        def d_gamma(lam: int, mu: int, nu: int, deriv: int) -> Tensor:
            return dgamma[..., lam * 16 + mu * 4 + nu, deriv]

        # Riemann: R^rho_{sigma mu nu}
        #        = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
        #        + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
        #        - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}
        def riemann(rho: int, sig: int, mu: int, nu: int) -> Tensor:
            lin = d_gamma(rho, nu, sig, mu) - d_gamma(rho, mu, sig, nu)
            quad = None
            for lam in range(4):
                t = gamma(rho, mu, lam) * gamma(lam, nu, sig) - gamma(rho, nu, lam) * gamma(lam, mu, sig)
                quad = t if quad is None else quad + t
            return lin + quad

        # Ricci: R_{sigma nu} = sum_rho R^rho_{sigma rho nu}
        ricci = [[None] * 4 for _ in range(4)]
        for sig in range(4):
            for nu in range(4):
                s = None
                for rho in range(4):
                    t = riemann(rho, sig, rho, nu)
                    s = t if s is None else s + t
                ricci[sig][nu] = s

        # Ricci scalar R = g^{sigma nu} R_{sigma nu}
        ricci_scalar = None
        for sig in range(4):
            for nu in range(4):
                t = ginv[sig][nu] * ricci[sig][nu]
                ricci_scalar = t if ricci_scalar is None else ricci_scalar + t

        # Einstein tensor + cosmological term, vs. 8*pi*T (vacuum default).
        residuals = []
        for (mu, nu) in _G_PAIRS:
            g_munu = g_list[_G_PAIR_INDEX[(mu, nu)]]
            einstein_munu = ricci[mu][nu] - 0.5 * g_munu * ricci_scalar
            lhs = einstein_munu + self.Lambda * g_munu
            if self.stress_energy_fn is not None:
                T_munu = self.stress_energy_fn(points)[..., _G_PAIR_INDEX[(mu, nu)]]
            else:
                T_munu = b.zeros(g_munu.shape)
            residuals.append(lhs - 8.0 * math.pi * T_munu)

        return b.stack(residuals, axis=-1)  # (..., 10)


# ---------------------------------------------------------------------------
# 42. Phonons — dispersive lattice wave equation (continuum limit)
# ---------------------------------------------------------------------------

class PhononResidual(PDEResidual):
    r"""
    Continuum limit of a 1-D harmonic lattice chain (mass ``m``, spring
    constant ``K``, lattice spacing ``a``). Taylor-expanding the discrete
    chain equation m u_n'' = K(u_{n+1} - 2u_n + u_{n-1}) in ``a`` gives, to
    leading dispersive order:

        u_tt - c^2 u_xx - beta * a^2 * u_xxxx = 0,      c^2 = K a^2 / m,
                                                          beta = c^2 / 12

    The extra 4th-order term is what makes this a *phonon* (dispersive)
    wave equation rather than the non-dispersive ``WaveResidual`` already
    in this file: it reproduces the acoustic-branch dispersion relation
    omega(k) = 2*sqrt(K/m)*|sin(k a / 2)| to O(k^4) instead of assuming a
    constant group velocity at all wavenumbers.

    ``model_fn(xt)`` -> [u]   (lattice displacement field)
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        lattice_spacing: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(
            backend, mass=mass, spring_constant=spring_constant,
            lattice_spacing=lattice_spacing, diff_mode=diff_mode,
        )
        self.m = mass
        self.K = spring_constant
        self.a = lattice_spacing
        self.c_sq = spring_constant * lattice_spacing ** 2 / mass
        self.beta = self.c_sq / 12.0

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        u_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731

        u_tt = _time_deriv(
            b,
            lambda x: _time_deriv(b, u_fn, x, t_index=-1, mode=self.diff_mode),
            points, t_index=-1, mode=self.diff_mode,
        )

        # d^2u/dx^2 via nested single-column grad (spatial_dims=1 case of
        # the existing _laplacian is exactly this, reused as-is).
        u_xx = _laplacian(b, u_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        # d^4u/dx^4 = laplacian of u_xx w.r.t. x again (1-D spatial domain).
        def u_xx_fn(x: Tensor) -> Tensor:
            return _laplacian(b, u_fn, x, spatial_dims=x.shape[-1] - 1, mode=self.diff_mode)

        u_xxxx = _laplacian(b, u_xx_fn, points, spatial_dims=points.shape[-1] - 1, mode=self.diff_mode)

        return u_tt - self.c_sq * u_xx - self.beta * self.a ** 2 * u_xxxx


# ---------------------------------------------------------------------------
# 43. Dirac Equation (1+1-D, 2-component spinor)
# ---------------------------------------------------------------------------

class DiracResidual(PDEResidual):
    r"""
    Dirac equation in 1+1 dimensions, H = c*sigma_x*p + m*c^2*sigma_z:

        i*hbar dpsi/dt = c*sigma_x*(-i*hbar*d/dx)*psi + m*c^2*sigma_z*psi

    reduced (dimensionally, following the same honest 1-D-core precedent
    already used by ``RelativisticFluidCoreResidual`` in this file — full
    3+1-D Dirac needs a 4-component spinor and the full gamma-matrix
    algebra, which is a real follow-up extension, not this class) to a
    2-component complex spinor psi = (psi1, psi2), split into 4 real
    fields for the network output since gamma matrices are complex:

        model_fn(xt) -> [u1, v1, u2, v2]   where psi1 = u1 + i v1,
                                                   psi2 = u2 + i v2

    giving the coupled real system (hbar=1 by default, rescale c/m for
    other unit conventions):

        hbar*du1/dt + hbar*c*du2/dx - m*c^2*v1 = 0
        hbar*dv1/dt + hbar*c*dv2/dx + m*c^2*u1 = 0
        hbar*du2/dt + hbar*c*du1/dx + m*c^2*v2 = 0
        hbar*dv2/dt + hbar*c*dv1/dx - m*c^2*u2 = 0

    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mass: float = 1.0,
        speed_of_light: float = 1.0,
        hbar: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, mass=mass, c=speed_of_light, hbar=hbar, diff_mode=diff_mode)
        self.m = mass
        self.c = speed_of_light
        self.hbar = hbar

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        mc2 = self.m * self.c ** 2

        def comp(idx: int) -> Callable:
            return lambda x: model_fn(x)[..., idx]

        u1_fn, v1_fn, u2_fn, v2_fn = comp(0), comp(1), comp(2), comp(3)

        def dt(fn: Callable) -> Tensor:
            return _time_deriv(b, fn, points, t_index=-1, mode=self.diff_mode)

        def dx(fn: Callable) -> Tensor:
            return _spatial_grad(b, fn, points, space_axes=[0], mode=self.diff_mode)[..., 0]

        u1, v1 = u1_fn(points), v1_fn(points)
        u2, v2 = u2_fn(points), v2_fn(points)

        res1 = self.hbar * dt(u1_fn) + self.hbar * self.c * dx(u2_fn) - mc2 * v1
        res2 = self.hbar * dt(v1_fn) + self.hbar * self.c * dx(v2_fn) + mc2 * u1
        res3 = self.hbar * dt(u2_fn) + self.hbar * self.c * dx(u1_fn) + mc2 * v2
        res4 = self.hbar * dt(v2_fn) + self.hbar * self.c * dx(v1_fn) - mc2 * u2

        return b.stack([res1, res2, res3, res4], axis=-1)


# ---------------------------------------------------------------------------
# 44. Quantum Gases — Bose-Einstein (Gross-Pitaevskii with trap potential)
# ---------------------------------------------------------------------------

class BoseEinsteinCondensateResidual(PDEResidual):
    r"""
    Gross-Pitaevskii equation for a trapped Bose-Einstein condensate:

        i*hbar dpsi/dt = -hbar^2/(2m) lap(psi) + V_trap(x)*psi + g|psi|^2*psi

    This is the defining mean-field PDE of BEC physics; what distinguishes
    it from the plain ``NLSResidual`` already in this file is the explicit
    external trapping potential ``trap_fn`` (harmonic trap by default),
    without which there is no *condensate* to speak of — just a free
    nonlinear Schroedinger pulse.

    ``model_fn(xt)`` -> [u, v]   where psi = u + i v
    ``points`` columns: [x, ..., t]  (last column is time; all others
    spatial, so this supports 1-D/2-D/3-D traps without changes)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mass: float = 1.0,
        hbar: float = 1.0,
        interaction_g: float = 1.0,
        trap_omega: float = 1.0,
        trap_fn: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(
            backend, mass=mass, hbar=hbar, interaction_g=interaction_g,
            trap_omega=trap_omega, diff_mode=diff_mode,
        )
        self.m = mass
        self.hbar = hbar
        self.g = interaction_g
        self.omega = trap_omega
        # Default: isotropic harmonic trap V(x) = 1/2 m omega^2 |x|^2
        self.trap_fn = trap_fn or self._default_harmonic_trap

    def _default_harmonic_trap(self, points: Tensor) -> Tensor:
        b = self.backend
        spatial = points[..., :-1]
        r_sq = b.sum(spatial * spatial, axis=-1)
        return 0.5 * self.m * self.omega ** 2 * r_sq

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend
        spatial_dims = points.shape[-1] - 1

        u_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731
        v_fn = lambda x: model_fn(x)[..., 1]  # noqa: E731

        u, v = u_fn(points), v_fn(points)
        u_t = _time_deriv(b, u_fn, points, mode=self.diff_mode)
        v_t = _time_deriv(b, v_fn, points, mode=self.diff_mode)
        lap_u = _laplacian(b, u_fn, points, spatial_dims=spatial_dims, mode=self.diff_mode)
        lap_v = _laplacian(b, v_fn, points, spatial_dims=spatial_dims, mode=self.diff_mode)

        V = self.trap_fn(points)
        prob = u * u + v * v  # |psi|^2

        # Real/imag split of i*hbar*psi_t = -hbar^2/(2m) lap(psi) + V*psi + g|psi|^2*psi
        # Real part  (from i*hbar*(u_t + i v_t) matched to RHS):
        res_real = -self.hbar * v_t + self.hbar ** 2 / (2 * self.m) * lap_u - V * u - self.g * prob * u
        res_imag = self.hbar * u_t + self.hbar ** 2 / (2 * self.m) * lap_v - V * v - self.g * prob * v

        return b.stack([res_real, res_imag], axis=-1)


# ---------------------------------------------------------------------------
# 45. Quantum Gases — Fermi-Dirac (degenerate Fermi gas hydrodynamics)
# ---------------------------------------------------------------------------

class FermiGasResidual(PDEResidual):
    r"""
    Hydrodynamic (Euler) equations for a degenerate, zero-temperature,
    non-relativistic Fermi gas, closed with the actual Fermi-Dirac
    equation of state for a free Fermi gas rather than an ideal-gas
    closure — this is what makes it Fermi-Dirac physics rather than a
    relabelled ``EulerResidual``:

        p(n) = K * n^(5/3),   K = (hbar^2 / 5m) * (6*pi^2 / g_s)^(2/3)

    (the standard degenerate-Fermi-gas polytrope, g_s = spin degeneracy;
    same closure used for e.g. white-dwarf and ultracold-Fermi-gas
    hydrodynamics). Continuity + momentum:

        dn/dt + d(n*u)/dx = 0
        d(n*u)/dt + d(n*u^2 + p(n))/dx = 0

    ``model_fn(xt)`` -> [n, u]   (number density, velocity)
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        mass: float = 1.0,
        hbar: float = 1.0,
        spin_degeneracy: float = 2.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(
            backend, mass=mass, hbar=hbar, spin_degeneracy=spin_degeneracy,
            diff_mode=diff_mode,
        )
        self.m = mass
        self.hbar = hbar
        self.g_s = spin_degeneracy
        self.K = (hbar ** 2 / (5.0 * mass)) * (6.0 * math.pi ** 2 / spin_degeneracy) ** (2.0 / 3.0)

    def _pressure(self, n: Tensor) -> Tensor:
        b = self.backend
        # n^(5/3) via exp((5/3) log n); density must stay positive, same
        # domain-validity reasoning already used for RelativisticFluidCoreResidual's
        # tanh velocity clamp — enforce positivity at the model-output level
        # (e.g. softplus) rather than fudging it inside the residual.
        return self.K * b.exp((5.0 / 3.0) * b.log(n))

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        n_fn = lambda x: model_fn(x)[..., 0]  # noqa: E731
        u_fn = lambda x: model_fn(x)[..., 1]  # noqa: E731

        def momentum_density_fn(x: Tensor) -> Tensor:
            return n_fn(x) * u_fn(x)

        def momentum_flux_fn(x: Tensor) -> Tensor:
            n_val, u_val = n_fn(x), u_fn(x)
            return n_val * u_val ** 2 + self._pressure(n_val)

        n_t = _time_deriv(b, n_fn, points, mode=self.diff_mode)
        mom_x = b.grad(lambda x: b.sum(momentum_density_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_continuity = n_t + mom_x

        mom_t = _time_deriv(b, momentum_density_fn, points, mode=self.diff_mode)
        flux_x = b.grad(lambda x: b.sum(momentum_flux_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_momentum = mom_t + flux_x

        return b.stack([res_continuity, res_momentum], axis=-1)


# ---------------------------------------------------------------------------
# 46. Quantum Relativistic Fluid (relativistic fluid + Bohm quantum potential)
# ---------------------------------------------------------------------------

class QuantumRelativisticFluidResidual(PDEResidual):
    r"""
    Extends ``RelativisticFluidCoreResidual`` with the Madelung/Bohm
    quantum-potential correction to the pressure closure:

        p_total(h, x) = p_classical(h) + Q(h),
        Q(h) = -(hbar^2 / 2m) * d^2(sqrt(h)) / dx^2 / sqrt(h)

    which is exactly what distinguishes a *quantum* relativistic fluid
    from the purely classical one already in this file: Q is the same
    quantum-pressure term that turns the classical Euler equations into
    the quantum hydrodynamic (Madelung) equations in the non-relativistic
    limit, here added on top of the relativistic energy-momentum
    conservation laws. Because Q involves d^2/dx^2 of the state, this
    residual is genuinely 2nd-order, unlike the 1st-order classical core.

    ``model_fn(xt)`` -> [h, u]  (enthalpy density profile, relative velocity)
    ``points`` columns: [x, t]
    """

    def __init__(
        self,
        backend: AbstractBackend,
        speed_of_light: float = 1.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, c=speed_of_light, mass=mass, hbar=hbar, diff_mode=diff_mode)
        self.c = speed_of_light
        self.c_sq = speed_of_light ** 2
        self.m = mass
        self.hbar = hbar

    def __call__(self, model_fn: Callable, points: Tensor) -> Tensor:
        b = self.backend

        # Same domain-safe velocity parameterisation as RelativisticFluidCoreResidual
        # (u = c*tanh(raw) is an exact bijection onto the admissible range, not
        # an approximation), applied identically here for the same reason.
        def relative_velocity(pt: Tensor) -> Tensor:
            raw_u = model_fn(pt)[..., 1]
            return self.c * b.tanh(raw_u)

        def h_fn(pt: Tensor) -> Tensor:
            return model_fn(pt)[..., 0]

        def sqrt_h_fn(pt: Tensor) -> Tensor:
            return b.sqrt(h_fn(pt))

        # Bohm quantum potential Q = -(hbar^2/2m) * (d^2 sqrt(h)/dx^2) / sqrt(h).
        # Needed both as a value (folded into the energy-equation's pressure
        # term below) and, via its own spatial derivative, as a force term in
        # the momentum equation — so it's wrapped as a function of the
        # collocation point rather than computed once, reusing the existing
        # _laplacian helper in its 1-spatial-dim form, unmodified.
        def quantum_potential_fn(pt: Tensor) -> Tensor:
            lap_sh = _laplacian(b, sqrt_h_fn, pt, spatial_dims=pt.shape[-1] - 1, mode=self.diff_mode)
            return -(self.hbar ** 2 / (2.0 * self.m)) * lap_sh / sqrt_h_fn(pt)

        def state_variables(pt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            h_val = h_fn(pt)
            u_val = relative_velocity(pt)
            lorentz = 1.0 / b.sqrt(1.0 - (u_val ** 2 / self.c_sq))
            p_total = 0.333 * h_val + quantum_potential_fn(pt)
            return h_val, u_val, lorentz * h_val - p_total

        energy_density_fn = lambda pt: state_variables(pt)[2]  # noqa: E731

        def momentum_density_fn(pt: Tensor) -> Tensor:
            h_val = h_fn(pt)
            u_val = relative_velocity(pt)
            lorentz_sq = 1.0 / (1.0 - (u_val ** 2 / self.c_sq))
            return lorentz_sq * h_val * u_val

        def momentum_flux_fn(pt: Tensor) -> Tensor:
            h_val = h_fn(pt)
            u_val = relative_velocity(pt)
            lorentz_sq = 1.0 / (1.0 - (u_val ** 2 / self.c_sq))
            p_total = 0.333 * h_val + quantum_potential_fn(pt)
            return lorentz_sq * h_val * u_val ** 2 + p_total

        energy_t = _time_deriv(b, energy_density_fn, points, mode=self.diff_mode)
        momentum_flux_x = b.grad(lambda x: b.sum(momentum_density_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_energy = energy_t + momentum_flux_x

        momentum_t = _time_deriv(b, momentum_density_fn, points, mode=self.diff_mode)
        stress_x = b.grad(lambda x: b.sum(momentum_flux_fn(x)), mode=self.diff_mode)(points)[..., 0]
        res_momentum = momentum_t + stress_x

        return b.stack([res_energy, res_momentum], axis=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PDE_REGISTRY: Dict[str, type] = {
    "poisson":           PoissonResidual,
    "laplace":           PoissonResidual,
    "heat":              HeatResidual,
    "diffusion":         HeatResidual,
    "wave":              WaveResidual,
    "burgers":           BurgersResidual,
    "navier_stokes":     NavierStokesResidual,
    "advection":         AdvectionResidual,
    "helmholtz":         HelmholtzResidual,
    "allen_cahn":        AllenCahnResidual,
    "cahn_hilliard":     CahnHilliardResidual,
    "schrodinger":       SchrodingerResidual,
    "klein_gordon":      KleinGordonResidual,
    "reaction_diffusion": ReactionDiffusionResidual,
    "eikonal":           EikonalResidual,
    "darcy":             DarcyResidual,
    "euler":             EulerResidual,
    "biharmonic":        BiharmonicResidual,
    "stokes":            StokesResidual,
    "nls":               NLSResidual,
    "kdv":               KdVResidual,
    "fokker_planck":     FokkerPlanckResidual,
    "nlse_2d":                   NonlinearSchrodinger2DResidual,
    "sine_gordon_2d":            SineGordon2DResidual,
    "fisher_kpp":                FisherKPPResidual,
    "klein_gordon_nonlinear_2d": KleinGordonNonlinear2DResidual,
    "fitzhugh_nagumo":           FitzHughNagumoResidual,
    "cahn_hilliard_2d":          CahnHilliard2DResidual,
    "kuramoto_sivashinsky":      KuramotoSivashinskyResidual,
    "kadomtsev_petviashvili":    KadomtsevPetviashviliResidual,
    "porous_medium":             PorousMediumResidual,
    "biharmonic_steady":         BiharmonicSteadyResidual,
    "boussinesq_wave":           BoussinesqWaveResidual,
    "euler_tricomi":             EulerTricomiResidual,
    "fokker_planck_2d":          FokkerPlanck2DResidual,
    "swift_hohenberg":           SwiftHohenbergResidual,
    "black_scholes_2d":          BlackScholes2DResidual,
    "regularised_long_wave":     RegularisedLongWaveResidual,
    "gierer_meinhardt":          GiererMeinhardtResidual,
    "gray_scott":                GrayScottResidual,
    "viscous_wave":              ViscousWaveResidual,
    "radhakrishnan_kundu_lakshmanan": RadhakrishnanKunduLakshmananResidual,
    "complex_ginzburg_landau_2d": ComplexGinzburgLandau2DResidual,
    "drift_diffusion_poisson":   DriftDiffusionPoissonResidual,
    "shallow_water_2d":          ShallowWater2DResidual,
    "heston_volatility":         HestonVolatilityResidual,
    "phase_field_crystal":       PhaseFieldCrystalResidual,
    "brinkman_darcy":            BrinkmanDarcyFlowResidual,
    "perona_malik":              PeronaMalikResidual,
    "burgers_2d":                Burgers2DResidual,
    "boussinesq_convection":     BoussinesqConvectionResidual,
    "dendritic_solidification":   DendriticSolidificationResidual,
    "relativistic_fluid":        RelativisticFluidCoreResidual,
    "einstein_field":            EinsteinFieldResidual,
    "efe":                       EinsteinFieldResidual,
    "phonon":                    PhononResidual,
    "phonons":                   PhononResidual,
    "dirac":                     DiracResidual,
    "dirac_equation":            DiracResidual,
    "bose_einstein":             BoseEinsteinCondensateResidual,
    "gross_pitaevskii":          BoseEinsteinCondensateResidual,
    "fermi_gas":                 FermiGasResidual,
    "fermi_dirac":                FermiGasResidual,
    "quantum_relativistic_fluid": QuantumRelativisticFluidResidual,
}


# ---------------------------------------------------------------------------
# Mixed / composite residuals — coupled multi-physics systems
# ---------------------------------------------------------------------------

class MixedResidual(PDEResidual):
    """
    Combine several ``PDEResidual`` terms — each possibly acting on a
    different slice of a shared model output — into a single weighted
    residual/loss, for coupled multi-physics systems (e.g. a Schrödinger
    field coupled to a Poisson potential, or separate Maxwell field
    components sharing collocation points).

    Two usage patterns are supported:

    1. **Shared output, field slices.** Pass ``field_slices`` mapping each
       named component to a slice/index of the model's output so each
       sub-residual only sees its own field(s), e.g. a model outputting
       ``[psi_real, psi_imag, V]`` for a Schrödinger-Poisson system.
    2. **Independent model outputs.** Omit ``field_slices`` and each
       sub-residual receives the full ``model_fn`` output — appropriate
       when each residual already knows how to index the fields it needs.

    Parameters
    ----------
    terms : mapping of name -> (PDEResidual instance, weight). Weights
        default to 1.0 if a bare residual (no tuple) is given.
    field_slices : optional mapping of name -> slice/int selecting the
        output column(s) that residual's ``model_fn`` should see.
    diff_mode : propagated to any sub-residual that does not already set
        its own ``diff_mode`` explicitly (does not override one that was
        already configured on construction).

    Examples
    --------
    >>> schrodinger = SchrodingerResidual(backend, hbar=1.0, m=1.0)
    >>> poisson     = PoissonResidual(backend)
    >>> mixed = MixedResidual(
    ...     backend,
    ...     terms={"schrodinger": (schrodinger, 1.0), "poisson": (poisson, 0.1)},
    ...     field_slices={"schrodinger": slice(0, 2), "poisson": slice(2, 3)},
    ... )
    >>> loss = mixed.loss(model_fn, collocation_points)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        terms: Dict[str, Any],
        field_slices: Optional[Dict[str, Any]] = None,
        *,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.terms: Dict[str, Tuple[PDEResidual, float]] = {
            name: (spec if isinstance(spec, tuple) else (spec, 1.0))
            for name, spec in terms.items()
        }
        self.field_slices = field_slices or {}

    def _sub_model_fn(
        self,
        name: str,
        model_fn: Callable[[Tensor], Tensor],
    ) -> Callable[[Tensor], Tensor]:
        sl = self.field_slices.get(name)
        if sl is None:
            return model_fn

        def sliced_fn(x: Tensor) -> Tensor:
            out = model_fn(x)
            return out[..., sl] if isinstance(sl, slice) else out[..., sl : sl + 1]

        return sliced_fn

    def residuals(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Dict[str, Tensor]:
        """Return each named sub-residual's raw (unweighted) tensor."""
        out: Dict[str, Tensor] = {}
        for name, (residual, _weight) in self.terms.items():
            sub_fn = self._sub_model_fn(name, model_fn)
            out[name] = residual(sub_fn, points)
        return out

    def __call__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Unlike single-field residuals, ``MixedResidual`` returns a dict of
        per-term residual tensors rather than one tensor — callers that need
        a single scalar should use ``.loss(...)``, which combines them with
        the configured weights.
        """
        return self.residuals(model_fn, points)

    def loss(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Tensor:
        """Weighted sum of each sub-residual's MSE."""
        b = self.backend
        total = None
        for name, (residual, weight) in self.terms.items():
            sub_fn = self._sub_model_fn(name, model_fn)
            r = residual(sub_fn, points)
            term_loss = weight * b.mean(b.square(r))
            total = term_loss if total is None else total + term_loss
        return total

    def loss_dict(
        self,
        model_fn: Callable[[Tensor], Tensor],
        points: Tensor,
    ) -> Dict[str, Tensor]:
        """Per-term weighted MSE losses, useful for logging/adaptive balancing."""
        b = self.backend
        out: Dict[str, Tensor] = {}
        for name, (residual, weight) in self.terms.items():
            sub_fn = self._sub_model_fn(name, model_fn)
            r = residual(sub_fn, points)
            out[name] = weight * b.mean(b.square(r))
        return out


# ---------------------------------------------------------------------------
# User-extensible registry: register_pde / unregister_pde
# ---------------------------------------------------------------------------

_REGISTRY_LOCK = threading.RLock()
_PDE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Snapshot of the shipped equations. Built-ins can only be replaced with an
#: explicit ``overwrite=True`` and can always be restored by ``unregister_pde``.
_BUILTIN_PDE_CLASSES: Dict[str, type] = dict(PDE_REGISTRY)
_BUILTIN_PDE_NAMES = frozenset(_BUILTIN_PDE_CLASSES)
_BUILTIN_META_BACKUP: Dict[str, Any] = {}   # key -> original PDEMeta (set on first overwrite)
_USER_PDE_NAMES: set = set()


def _normalise_pde_name(name: Any, what: str = "name") -> str:
    if not isinstance(name, str):
        raise TypeError(f"PDE {what} must be a str, got {type(name).__name__}.")
    key = name.strip().lower()
    if not _PDE_NAME_RE.match(key):
        raise ValueError(
            f"Invalid PDE {what} {name!r}: use letters, digits and "
            "underscores only, starting with a letter (e.g. 'my_pde'). "
            "Names are case-insensitive."
        )
    return key


def _validate_residual_class(cls: Any) -> None:
    if not isinstance(cls, type):
        raise TypeError(
            f"register_pde expects a class, got {type(cls).__name__}. "
            "Pass the class itself (not an instance): register_pde('x', MyResidual)."
        )
    if not issubclass(cls, PDEResidual):
        raise TypeError(
            f"{cls.__name__} must subclass physai.PDEResidual. If you only "
            "have a plain function (model_fn, points) -> residual, pass it "
            "straight to Trainer(residual=...) instead of registering it."
        )
    if cls is PDEResidual or inspect.isabstract(cls):
        raise TypeError(f"{cls.__name__} is abstract; register a concrete subclass.")
    if cls.__call__ is PDEResidual.__call__:
        raise TypeError(
            f"{cls.__name__} must override __call__(self, model_fn, points) "
            "and return the residual tensor."
        )
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return  # signature not introspectable; trust the class
    try:
        sig.bind(None)  # build_residual calls cls(backend, **kwargs)
    except TypeError:
        raise TypeError(
            f"{cls.__name__}.__init__ must accept the backend as its first "
            f"positional argument (build_residual calls cls(backend, **kwargs)); "
            f"got signature {sig}."
        ) from None


def _coerce_pde_meta(key: str, meta: Any, cls: type, existing: Any, ao: Any) -> Any:
    """Turn user input (PDEMeta | dict | None) into a validated PDEMeta for ``key``."""
    if meta is None:
        meta = getattr(cls, "PDE_META", None)
    if meta is None:
        meta = existing                      # keep prior metadata when overwriting
    if meta is None:
        meta = dict(ao.PDE_META_DEFAULTS)    # generic defaults, no warning later
    if isinstance(meta, dict):
        data = {**ao.PDE_META_DEFAULTS, **meta}
        data.pop("name", None)
        try:
            meta = ao.PDEMeta(name=key, **data)
        except TypeError as exc:
            raise TypeError(f"Invalid meta dict for '{key}': {exc}") from None
    elif isinstance(meta, ao.PDEMeta):
        meta = _dc_replace(meta, name=key)
    else:
        raise TypeError(
            f"meta must be a PDEMeta, a dict of PDEMeta fields, or None; "
            f"got {type(meta).__name__}."
        )
    ao._validate_pde_meta(meta)
    return meta


def register_pde(
    name: str,
    cls: Optional[type] = None,
    *,
    meta: Any = None,
    aliases: Sequence[str] = (),
    overwrite: bool = False,
):
    """Register a user-defined PDE residual so it works by *name* everywhere.

    After registration ``build_residual(name, backend, **params)``,
    ``ProblemSpec(pde_name=name, ...)``, ``AutoOptimizer`` and residual-
    adaptive refinement all resolve it exactly like a built-in equation.

    Use as a function or a decorator::

        class MyPDE(physai.PDEResidual):
            def __call__(self, model_fn, points):
                u_t = _time_deriv(self.backend, model_fn, points, mode=self.diff_mode)
                ...
                return u_t - self.params.get("k", 1.0) * lap

        physai.register_pde("my_pde", MyPDE, meta={"order": 2, "nonlinear": False})

        @physai.register_pde("my_pde2", meta={"order": 4})
        class MyPDE2(physai.PDEResidual): ...

    Parameters
    ----------
    name : case-insensitive identifier (letters, digits, ``_``; starts with a letter).
    cls : concrete ``PDEResidual`` subclass that overrides ``__call__`` and whose
        constructor accepts ``backend`` as first positional argument. Omit to use
        as a decorator.
    meta : optional hints for AutoOptimizer -- a ``PDEMeta``, a dict with any of
        ``order, nonlinear, n_components, stiff, spectral_bias ("low"|"mixed"|"high"),
        recommended_arch ("pinn"|"fno"|"both")``, or None. Resolution order when
        None: ``cls.PDE_META`` attribute, then the existing metadata (when
        overwriting), then generic defaults (order 2, linear, 1 field, "pinn").
    aliases : extra names resolving to the same class (each gets its own meta entry).
    overwrite : allow replacing an existing name. Required for shipped equations.
        Re-running a notebook cell that redefines *your own* class (same module and
        qualname) is accepted without it.

    The call is atomic: on any error nothing is registered. Returns ``cls``.
    """
    if cls is None:
        if isinstance(name, type):
            raise TypeError(
                "Use register_pde('name', Class) or @register_pde('name'); "
                "the name is required."
            )
        # Fail fast on a bad name/aliases before the decorated class is seen.
        _normalise_pde_name(name)
        for a in ((aliases,) if isinstance(aliases, str) else aliases):
            _normalise_pde_name(a, "alias")

        def _decorator(c: type) -> type:
            return register_pde(name, c, meta=meta, aliases=aliases, overwrite=overwrite)

        return _decorator

    if isinstance(aliases, str):
        aliases = (aliases,)
    keys = [_normalise_pde_name(name)] + [_normalise_pde_name(a, "alias") for a in aliases]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate names in {keys}.")
    _validate_residual_class(cls)

    from physai.core import auto_optimizer as ao  # lazy: heavy, and avoids import cycles

    with _REGISTRY_LOCK:
        # ---- phase 1: validate everything, mutate nothing --------------------
        plan = []
        for key in keys:
            old_cls = PDE_REGISTRY.get(key)
            if old_cls is not None and not overwrite:
                same_user_class = (
                    key not in _BUILTIN_PDE_NAMES
                    and key in _USER_PDE_NAMES
                    and (old_cls is cls or (
                        old_cls.__module__ == cls.__module__
                        and old_cls.__qualname__ == cls.__qualname__
                    ))
                )
                if not same_user_class:
                    origin = "built-in" if key in _BUILTIN_PDE_NAMES else "already registered"
                    raise ValueError(
                        f"PDE '{key}' is {origin} ({old_cls.__name__}). "
                        "Pass overwrite=True to replace it, or pick another name."
                    )
            old_meta = ao.get_pde_meta(key)
            plan.append((key, old_cls, old_meta, _coerce_pde_meta(key, meta, cls, old_meta, ao)))

        # ---- phase 2: commit, roll back on any failure ------------------------
        done = []
        try:
            for key, old_cls, old_meta, new_meta in plan:
                if key in _BUILTIN_PDE_NAMES and key not in _BUILTIN_META_BACKUP:
                    _BUILTIN_META_BACKUP[key] = old_meta
                PDE_REGISTRY[key] = cls
                ao.register_pde_meta(new_meta, overwrite=True)
                done.append((key, old_cls, old_meta))
        except BaseException:
            for key, old_cls, old_meta in reversed(done):
                if old_cls is None:
                    PDE_REGISTRY.pop(key, None)
                else:
                    PDE_REGISTRY[key] = old_cls
                if old_meta is None:
                    ao.unregister_pde_meta(key)
                else:
                    ao.register_pde_meta(old_meta, overwrite=True)
            raise
        for key, _, _, _ in plan:
            if key not in _BUILTIN_PDE_NAMES:
                _USER_PDE_NAMES.add(key)
    return cls


def unregister_pde(name: str, *, missing_ok: bool = False) -> None:
    """Undo ``register_pde`` for ``name`` (registry entry and metadata).

    * user-registered name -> removed;
    * shipped equation you replaced with ``overwrite=True`` -> the original
      class and metadata are restored;
    * untouched shipped equation -> ``ValueError`` (built-ins can't be removed).
    """
    key = _normalise_pde_name(name)
    from physai.core import auto_optimizer as ao

    with _REGISTRY_LOCK:
        if key in _BUILTIN_PDE_NAMES:
            if PDE_REGISTRY.get(key) is _BUILTIN_PDE_CLASSES[key]:
                raise ValueError(f"'{key}' is a built-in PDE and cannot be unregistered.")
            PDE_REGISTRY[key] = _BUILTIN_PDE_CLASSES[key]
            orig = _BUILTIN_META_BACKUP.pop(key, None)
            if orig is not None:
                ao.register_pde_meta(orig, overwrite=True)
            return
        if key not in PDE_REGISTRY:
            if missing_ok:
                ao.unregister_pde_meta(key)
                return
            raise KeyError(f"PDE '{key}' is not registered.")
        del PDE_REGISTRY[key]
        _USER_PDE_NAMES.discard(key)
        ao.unregister_pde_meta(key)


def build_residual(
    name: str,
    backend: AbstractBackend,
    **kwargs: Any,
) -> PDEResidual:
    """
    Factory: instantiate a PDE residual by name.

    >>> r = build_residual("burgers", backend, nu=0.01)
    >>> loss = r.loss(model_fn, collocation_points)
    """
    if not isinstance(name, str):
        raise TypeError(f"PDE name must be a str, got {type(name).__name__}.")
    key = name.strip().lower()
    cls = PDE_REGISTRY.get(key)
    if cls is None:
        close = difflib.get_close_matches(key, list(PDE_REGISTRY), n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise ValueError(
            f"Unknown PDE '{name}'.{hint} Available: {list(PDE_REGISTRY)}. "
            "Custom equations: physai.register_pde(name, cls)."
        )
    return cls(backend, **kwargs)

__all__ = [
    # Core Structures & Config Maps
    "PDEResidual",
    "MixedResidual",
    "PDE_REGISTRY",
    "build_residual",
    "register_pde",
    "unregister_pde",
    
    # Original Systems
    "PoissonResidual",
    "HeatResidual",
    "WaveResidual",
    "BurgersResidual",
    "NavierStokesResidual",
    "AdvectionResidual",
    "HelmholtzResidual",
    "AllenCahnResidual",
    "CahnHilliardResidual",
    "SchrodingerResidual",
    "KleinGordonResidual",
    "ReactionDiffusionResidual",
    "EikonalResidual",
    "DarcyResidual",
    "EulerResidual",
    "BiharmonicResidual",
    "StokesResidual",
    "NLSResidual",
    "KdVResidual",
    "FokkerPlanckResidual",
    
    # New Suite: Advanced Research & Industrial Systems
    "NonlinearSchrodinger2DResidual",
    "SineGordon2DResidual",
    "FisherKPPResidual",
    "KleinGordonNonlinear2DResidual",
    "FitzHughNagumoResidual",
    "CahnHilliard2DResidual",
    "KuramotoSivashinskyResidual",
    "KadomtsevPetviashviliResidual",
    "PorousMediumResidual",
    "BiharmonicSteadyResidual",
    "BoussinesqWaveResidual",
    "EulerTricomiResidual",
    "FokkerPlanck2DResidual",
    "SwiftHohenbergResidual",
    "BlackScholes2DResidual",
    "RegularisedLongWaveResidual",
    "GiererMeinhardtResidual",
    "GrayScottResidual",
    "ViscousWaveResidual",
    "RadhakrishnanKunduLakshmananResidual",
    "ComplexGinzburgLandau2DResidual",
    "DriftDiffusionPoissonResidual",
    "ShallowWater2DResidual",
    "HestonVolatilityResidual",
    "PhaseFieldCrystalResidual",
    "BrinkmanDarcyFlowResidual",
    "PeronaMalikResidual",
    "Burgers2DResidual",
    "BoussinesqConvectionResidual",
    "DendriticSolidificationResidual",
    "RelativisticFluidCoreResidual",
    "EinsteinFieldResidual",
    "PhononResidual",
    "DiracResidual",
    "BoseEinsteinCondensateResidual",
    "FermiGasResidual",
    "QuantumRelativisticFluidResidual",
]