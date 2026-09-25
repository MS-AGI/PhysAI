"""
tests/test_latex_to_residual.py

Tests for the LaTeX -> residual compilation pipeline only
(physai.core.latex_pde: build_latex_residual / register_latex_pde), the
same code path the PhysAI Workbench web UI uses when a user writes a PDE
in LaTeX. This is where problems tend to surface: derivative-fraction
parsing, shorthand subscripts (u_xx), the \\nabla^2 / \\Delta macros,
nonlinear-term detection, multi-field indexing, and the safe-AST
whitelist that keeps user LaTeX from reaching Python's eval.

No training happens anywhere in this file — every test builds (or
registers) a residual, evaluates it on a hand-picked model_fn built from
a closed-form manufactured solution, and checks the residual is (near)
exactly zero. This is Tier 3 of tests/test_pde_residual.py's own scheme,
applied to the LaTeX front-end instead of the built-in PDE registry.

Three tiers, increasing in what they stress:
  EASY       — one field, one equation, textbook \\frac{\\partial ...}
               and shorthand notation, linear PDEs.
  MEDIUM     — nonlinear terms (u * u_x), coupled multi-field systems,
               \\nabla^2 combined with a parameter.
  DIFFICULT  — third-order derivatives + nonlinearity together (KdV
               soliton), mixed second partials (u_{xy}), and a 3-field
               system with a bilinear cross term and an auxiliary field
               that has no equation of its own.

A final section checks the pipeline's safety/registration behaviour:
that register_latex_pde -> build_residual(name, ...) (the actual path
buildScript() in the web app generates) agrees with build_latex_residual
called directly, and that malformed/unsafe LaTeX is rejected rather than
silently mishandled.

Run with: pytest tests/test_latex_to_residual.py -v
"""
import math

import numpy as np
import pytest
import torch

from physai.core.latex_pde import build_latex_residual, register_latex_pde
from physai.core.pde_residual import build_residual

# ---------------------------------------------------------------------------
# Backends: LatexPDEResidual is backend-agnostic (it only calls generic
# backend.grad / backend.sin / backend.stack / ... methods, nothing
# torch-specific), so every test below is parametrized across all four
# registered backends, skipping ones that aren't installed. This mirrors
# tests/test-backend/*_test.py's own skip-if-missing pattern rather than
# hand-picking one backend.
# ---------------------------------------------------------------------------

def _make_backend(name: str):
    if name == "torch":
        from physai.backends.torch_backend import TorchBackend
        return TorchBackend()
    if name == "jax":
        from physai.backends.jax_backend import JAXBackend
        return JAXBackend()
    if name == "tensorflow":
        from physai.backends.tensorflow_backend import TensorFlowBackend
        return TensorFlowBackend()
    if name == "paddle":
        from physai.backends.paddle_backend import PaddleBackend
        return PaddleBackend()
    raise ValueError(name)


def _available_backends():
    names = []
    for name in ("torch", "jax", "tensorflow", "paddle"):
        try:
            _make_backend(name)
            names.append(name)
        except ImportError:
            pass
    return names


BACKEND_NAMES = _available_backends()
if not BACKEND_NAMES:
    pytest.skip("no physai backend (torch/jax/tensorflow/paddle) is installed", allow_module_level=True)


@pytest.fixture(params=BACKEND_NAMES)
def backend(request):
    return _make_backend(request.param)


def _to_np(backend, t):
    return np.asarray(backend.to_numpy(t))


def _sin(backend, x): return backend.sin(x)
def _cos(backend, x): return backend.cos(x)
def _tanh(backend, x): return backend.tanh(x)
def _exp(backend, x): return backend.exp(x)
def _ones_like(backend, x): return backend.ones(x.shape) if hasattr(x, "shape") else backend.ones((1,))
def _stack(backend, arrs): return backend.stack(arrs, axis=-1)
# 1/(1+t), but built from log/exp rather than "/" or reciprocal: this
# installed Paddle build has no registered divide_double_grad or
# reciprocal_grad_grad kernel, so any *second*-order derivative through
# a plain division (e.g. u_xx of u = x/(1+t)) fails on that backend.
# log/exp both have fully registered double-grad kernels here (see
# paddle_backend.py's _pade_apply for the same trick), and the
# denominator is always >= 1.2 for the points these tests use, so this
# is exact, not an approximation.
def _reciprocal(backend, x): return backend.exp(-backend.log(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _points(backend, n: int, cols: int, seed: int = 0) -> torch.Tensor:
    """Random interior points, kept away from 0 so terms like 1/(1+t) or
    log(x) (used by a couple of manufactured solutions below) stay finite."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.2, 1.0, size=(n, cols)).astype(np.float32)
    return backend.tensor(arr)


def _assert_residual_zero(backend, r, atol: float = 1e-3) -> None:
    arr = _to_np(backend, r)
    assert np.isfinite(arr).all(), "residual contains NaN/Inf"
    assert np.allclose(arr, np.zeros_like(arr), atol=atol), (
        f"residual not ~0: max abs = {np.abs(arr).max():.3e}"
    )


def _assert_allclose(backend, a, b, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    """Backend-agnostic stand-in for torch.testing.assert_close, since a/b
    may be tensors from any of the four registered backends, not just torch."""
    np.testing.assert_allclose(_to_np(backend, a), _to_np(backend, b), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# EASY — single field, single equation, standard grammar
# ---------------------------------------------------------------------------

class TestEasy:
    def test_laplacian_macro_poisson_like(self, backend):
        r"""u = x^2 + y^2  =>  \nabla^2 u = 4, exactly.

        Exercises the \nabla^2 -> Lap(...) macro substitution and the
        2-D Laplacian sum.
        """
        residual = build_latex_residual(
            r"\nabla^2 u = 4",
            backend, fields=("u",), coordinates=("x", "y"),
        )
        pts = _points(backend, 8, 2, seed=1)
        model = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2)[..., None]  # noqa: E731
        r = residual(model, pts)
        _assert_residual_zero(backend, r)

    def test_heat_equation_via_frac_notation(self, backend):
        r"""u = exp(-alpha k^2 t) sin(k x) solves u_t = alpha*\nabla^2 u
        exactly for any alpha, k (homogeneous heat equation).

        Exercises \frac{\partial u}{\partial t}, the \nabla^2 macro, and
        a named parameter substituted into the compiled expression.
        """
        alpha, k = 0.7, 2.0
        residual = build_latex_residual(
            r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
            backend, fields=("u",), coordinates=("x", "t"),
            parameters={"alpha": alpha},
        )
        pts = _points(backend, 8, 2, seed=2)

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            u = _exp(backend, -alpha * k ** 2 * t) * _sin(backend, k * x)
            return u[..., None]

        r = residual(model, pts)
        _assert_residual_zero(backend, r)

    def test_heat_equation_shorthand_matches_frac_notation(self, backend):
        """The same PDE written with shorthand subscripts (u_t, u_xx)
        must compile to the same residual values as the \\frac version
        above — a regression check that both code paths through
        _latex_to_python agree."""
        alpha, k = 0.7, 2.0
        r_frac = build_latex_residual(
            r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
            backend, fields=("u",), coordinates=("x", "t"), parameters={"alpha": alpha},
        )
        r_short = build_latex_residual(
            "u_t = alpha*u_xx",
            backend, fields=("u",), coordinates=("x", "t"), parameters={"alpha": alpha},
        )
        pts = _points(backend, 8, 2, seed=2)

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            u = _exp(backend, -alpha * k ** 2 * t) * _sin(backend, k * x)
            return u[..., None]

        _assert_allclose(backend, r_frac(model, pts), r_short(model, pts))
        _assert_residual_zero(backend, r_short(model, pts))

    def test_pure_advection_traveling_wave(self, backend):
        """u = sin(x - c t) solves u_t + c u_x = 0 exactly for any c.
        Simplest possible transport equation; sanity floor for the tier."""
        c = 1.5
        residual = build_latex_residual(
            "u_t + c*u_x = 0",
            backend, fields=("u",), coordinates=("x", "t"), parameters={"c": c},
        )
        pts = _points(backend, 8, 2, seed=3)
        model = lambda xt: _sin(backend, xt[..., 0] - c * xt[..., 1])[..., None]  # noqa: E731
        r = residual(model, pts)
        _assert_residual_zero(backend, r)


# ---------------------------------------------------------------------------
# MEDIUM — nonlinear terms, coupled fields, macro + parameter together
# ---------------------------------------------------------------------------

class TestMedium:
    def test_inviscid_burgers_nonlinear_term(self, backend):
        r"""u = x / (1 + t) solves \partial_t u + u \partial_x u - \nu u_{xx} = 0
        exactly for *any* nu, since u_xx = 0 identically for this u:

            u_t  = -x/(1+t)^2
            u_x  =  1/(1+t)
            u*u_x =  x/(1+t)^2
            u_t + u*u_x = 0,  and u_xx = 0 kills the viscous term too.

        Exercises the nonlinear-product path (u * u_x, same field
        appearing in two multiplied subtrees) and the nonlinear-detection
        flag in one shot.
        """
        nu = 0.3
        residual = build_latex_residual(
            r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} "
            r"- \nu\frac{\partial^2 u}{\partial x^2} = 0",
            backend, fields=("u",), coordinates=("x", "t"), parameters={"nu": nu},
        )
        assert residual.nonlinear is True
        pts = _points(backend, 8, 2, seed=4)
        model = lambda xt: (xt[..., 0] * _reciprocal(backend, 1.0 + xt[..., 1]))[..., None]  # noqa: E731
        r = residual(model, pts)
        _assert_residual_zero(backend, r, atol=1e-2)  # nonlinear + 2nd-deriv autodiff noise

    def test_coupled_two_field_wave_system(self, backend):
        r"""First-order system for the wave equation:
            u_t - v        = 0
            v_t - c^2 u_xx = 0
        with u = sin(x - c t), v = u_t = -c cos(x - c t) — exact for any c.

        Exercises a multi-field model_fn (stacked [u, v] output), field
        selection by index inside D(...), and residual stacking across
        two equations (output axis=-1).
        """
        c = 1.2
        residual = build_latex_residual(
            [
                "u_t - v = 0",
                r"v_t - c^{2}\frac{\partial^2 u}{\partial x^2} = 0",
            ],
            backend, fields=("u", "v"), coordinates=("x", "t"), parameters={"c": c},
        )
        pts = _points(backend, 8, 2, seed=5)

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            u = _sin(backend, x - c * t)
            v = -c * _cos(backend, x - c * t)
            return _stack(backend, [u, v])

        r = residual(model, pts)
        assert r.shape[-1] == 2
        _assert_residual_zero(backend, r)

    def test_helmholtz_2d_laplacian_macro_with_parameter(self, backend):
        r"""u = sin(x) sin(y), k^2 = 2 solves \nabla^2 u + k^2 u = 0 exactly:
        Lap u = -2 sin(x) sin(y) = -k^2 u.

        Exercises the \nabla^2 macro combined with a squared parameter
        (k^{2}) in the same equation as a linear field term.
        """
        k = math.sqrt(2.0)
        residual = build_latex_residual(
            r"\nabla^2 u + k^{2} u = 0",
            backend, fields=("u",), coordinates=("x", "y"), parameters={"k": k},
        )
        pts = _points(backend, 8, 2, seed=6)
        model = lambda xy: (_sin(backend, xy[..., 0]) * _sin(backend, xy[..., 1]))[..., None]  # noqa: E731
        r = residual(model, pts)
        _assert_residual_zero(backend, r)


# ---------------------------------------------------------------------------
# DIFFICULT — 3rd-order + nonlinear together, mixed partials, N-field systems
# ---------------------------------------------------------------------------

class TestDifficult:
    def test_kdv_soliton_third_order_plus_nonlinear(self, backend):
        r"""Classic KdV 1-soliton for \partial_t u + 6 u u_x + u_{xxx} = 0:

            u(x, t) = (c/2) sech^2( (sqrt(c)/2)(x - c t) )

        sech^2(z) is written as 1 - tanh(z)^2 since the backend interface
        has tanh but not cosh/sech (mirrors the built-in KdV test in
        test_pde_residual.py). This is the deepest derivative chain in
        the suite (three nested D(...) calls via the \\partial^3 fraction
        path) stacked with the nonlinear 6*u*u_x term; float32 3rd-deriv
        autodiff noise needs a looser tolerance, same as the built-in
        registry's own KdV test.
        """
        residual = build_latex_residual(
            r"\frac{\partial u}{\partial t} + 6u\frac{\partial u}{\partial x} "
            r"+ \frac{\partial^3 u}{\partial x^3} = 0",
            backend, fields=("u",), coordinates=("x", "t"),
        )
        assert residual.nonlinear is True
        assert residual.order == 3
        pts = _points(backend, 10, 2, seed=7)
        c = 4.0
        sqrt_c = c ** 0.5

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            z = (sqrt_c / 2.0) * (x - c * t)
            tanh_z = _tanh(backend, z)
            sech2 = 1.0 - tanh_z * tanh_z
            u = (c / 2.0) * sech2
            return u[..., None]

        r = residual(model, pts)
        _assert_residual_zero(backend, r, atol=5e-2)

    def test_mixed_second_partial_shorthand(self, backend):
        """u = x*y solves u_{xy} = 1 exactly.

        Exercises the two-letter shorthand-subscript path (u_{xy} ->
        D(u, x, y)), i.e. a genuinely mixed (non-diagonal) second
        derivative rather than a repeated axis like u_xx.
        """
        residual = build_latex_residual(
            "u_{xy} = 1",
            backend, fields=("u",), coordinates=("x", "y"),
        )
        pts = _points(backend, 8, 2, seed=8)
        model = lambda xy: (xy[..., 0] * xy[..., 1])[..., None]  # noqa: E731
        r = residual(model, pts)
        _assert_residual_zero(backend, r)

    def test_three_field_bilinear_cross_term_with_auxiliary_field(self, backend):
        r"""A 3-field, 2-equation system with a bilinear (u*v) nonlinear
        cross term and a parameter, where the third field w has no
        equation of its own and only ever appears as a term in v's
        equation (an "auxiliary field" pattern: legal because _evaluate
        looks fields up by name, not by whether they own an equation).

            u_t - v                 = 0
            v_t - kappa*u*v - w     = 0

        Choosing u = t, v = 1 forces (from eq 1) v = u_t = 1 exactly, and
        then eq 2 pins down w = v_t - kappa*u*v = -kappa*t. Every term is
        then satisfied exactly by construction, for any kappa.

        Exercises 3-field indexing in a multi-output model_fn, a bilinear
        product of two *different* fields (as opposed to Medium's u*u_x
        self-product), and an unconstrained auxiliary field.
        """
        kappa = 0.8
        residual = build_latex_residual(
            [
                "u_t - v = 0",
                "v_t - kappa*u*v - w = 0",
            ],
            backend, fields=("u", "v", "w"), coordinates=("x", "t"), parameters={"kappa": kappa},
        )
        assert residual.nonlinear is True
        pts = _points(backend, 8, 2, seed=9)

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            u = t
            v = _ones_like(backend, t)
            w = -kappa * t
            return _stack(backend, [u, v, w])

        r = residual(model, pts)
        assert r.shape[-1] == 2
        _assert_residual_zero(backend, r)


# ---------------------------------------------------------------------------
# Registration path + safety — same buildScript() code path as the web app,
# and the "safe AST interpreter, no eval" guarantee the module docstring
# claims. Still no training.
# ---------------------------------------------------------------------------

class TestRegistrationAndSafety:
    def test_register_latex_pde_matches_direct_build(self, backend):
        """register_latex_pde(...) then build_residual(name, backend) —
        exactly what the web app's generated script does — must agree
        with calling build_latex_residual directly on the same equation."""
        name = "test_only_heat_via_registration"
        register_latex_pde(
            name, r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
            fields=("u",), coordinates=("x", "t"), parameters={"alpha": 0.5}, overwrite=True,
        )
        registered = build_residual(name, backend)
        direct = build_latex_residual(
            r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
            backend, fields=("u",), coordinates=("x", "t"), parameters={"alpha": 0.5},
        )
        pts = _points(backend, 6, 2, seed=10)

        def model(xt):
            x, t = xt[..., 0], xt[..., 1]
            return (_exp(backend, -0.5 * 4.0 * t) * _sin(backend, 2.0 * x))[..., None]

        _assert_allclose(backend, registered(model, pts), direct(model, pts))

    def test_unknown_symbol_is_rejected(self, backend):
        """A field/parameter/coordinate that was never declared must be
        rejected at compile time, not silently treated as e.g. 0."""
        with pytest.raises(ValueError):
            build_latex_residual(
                "u_t = beta*u_xx",  # 'beta' was never declared as a parameter
                backend, fields=("u",), coordinates=("x", "t"), parameters={"alpha": 1.0},
            )

    def test_disallowed_python_builtin_is_rejected_not_evaluated(self, backend):
        """The module promises a safe AST interpreter that never reaches
        eval(); a call to an arbitrary Python name (e.g. a builtin) must
        be rejected rather than silently executed or ignored."""
        with pytest.raises(ValueError):
            build_latex_residual(
                "u_t = __import__('os').system('echo unsafe')",
                backend, fields=("u",), coordinates=("x", "t"),
            )

    def test_malformed_derivative_fraction_is_rejected(self, backend):
        """A derivative fraction referencing a coordinate that was never
        declared must fail to compile instead of falling back to a wrong
        interpretation."""
        with pytest.raises(ValueError):
            build_latex_residual(
                r"\frac{\partial u}{\partial z} = 0",  # 'z' not in coordinates
                backend, fields=("u",), coordinates=("x", "t"),
            )

    def test_mismatched_point_columns_is_rejected(self, backend):
        """Feeding points with the wrong number of coordinate columns
        must raise, not silently broadcast or truncate."""
        residual = build_latex_residual(
            "u_t = u_xx", backend, fields=("u",), coordinates=("x", "t"),
        )
        bad_pts = _points(backend, 8, 3, seed=11)  # 3 columns, but only 2 coordinates declared
        model = lambda p: p[..., 0:1]  # noqa: E731
        with pytest.raises(ValueError):
            residual(model, bad_pts)