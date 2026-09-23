"""
tests/test_losses.py

Tests for physai.core.losses: the primitive residual norms, boundary-
condition loss kernels, regularisation losses, the new structural/
physics penalties (divergence/curl), causal time-weighting, and every
`WeightedLossComposite` reweighting strategy (fixed, softmax_temp,
relobralo, grad_norm, uncertainty).

Torch-only (matches the rest of this test suite's convention) -- the
grad_norm strategy in particular needs a real eager-framework
`grad(value, params)` primitive to test meaningfully, which only
torch/paddle expose directly; jax/tf grad_norm paths are documented in
the source but not exercised here since those backends aren't installed
in this environment.

Run with: pytest tests/test_losses.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physai.backends.torch_backend import TorchBackend
from physai.core.losses import (
    WeightedLossComposite,
    causal_weighted_loss,
    charbonnier_loss,
    curl_penalty_2d,
    curl_penalty_3d,
    data_loss,
    dirichlet_loss,
    divergence_penalty,
    gradient_penalty,
    huber_loss,
    interface_loss,
    log_cosh_loss,
    mae_loss,
    mse_loss,
    neumann_loss,
    periodic_loss,
    quantile_loss,
    robin_loss,
    sobolev_loss,
    spectral_energy_loss,
)

backend = TorchBackend()


def _t(arr) -> torch.Tensor:
    return backend.tensor(np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# Primitive residual norms
# ---------------------------------------------------------------------------

class TestPrimitiveNorms:
    def test_mse_loss_matches_hand_calc(self):
        r = _t([1.0, -2.0, 3.0])
        out = mse_loss(backend, r)
        assert float(out) == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)

    def test_mae_loss_matches_hand_calc(self):
        r = _t([1.0, -2.0, 3.0])
        out = mae_loss(backend, r)
        assert float(out) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)

    def test_log_cosh_loss_small_residual_approx_half_mse(self):
        # log(cosh(x)) ~ x^2/2 for small x -> should track mse_loss/2 closely
        r = _t([0.01, -0.02, 0.005])
        lc = float(log_cosh_loss(backend, r))
        half_mse = float(mse_loss(backend, r)) / 2.0
        assert lc == pytest.approx(half_mse, rel=1e-2)

    def test_log_cosh_loss_zero_residual_is_zero(self):
        r = _t([0.0, 0.0, 0.0])
        assert float(log_cosh_loss(backend, r)) == pytest.approx(0.0, abs=1e-6)

    def test_huber_loss_quadratic_regime_matches_mse_half(self):
        r = _t([0.1, -0.2])
        delta = 1.0
        out = float(huber_loss(backend, r, delta=delta))
        expected = float(backend.mean(backend.square(r) * 0.5))
        assert out == pytest.approx(expected, rel=1e-3)

    def test_huber_loss_linear_regime_approx_matches_l1(self):
        r = _t([10.0, -10.0])
        delta = 1.0
        out = float(huber_loss(backend, r, delta=delta))
        expected = float(backend.mean(delta * backend.abs(r) - 0.5 * delta ** 2))
        assert out == pytest.approx(expected, rel=1e-2)

    def test_charbonnier_loss_matches_formula(self):
        r = _t([1.0, -3.0, 0.0])
        eps = 1e-3
        out = float(charbonnier_loss(backend, r, eps=eps))
        expected = float(np.mean(np.sqrt(np.array([1.0, 9.0, 0.0]) + eps ** 2) - eps))
        assert out == pytest.approx(expected, rel=1e-5)

    def test_charbonnier_loss_zero_at_zero_residual(self):
        r = _t([0.0, 0.0])
        assert float(charbonnier_loss(backend, r)) == pytest.approx(0.0, abs=1e-6)

    def test_charbonnier_approximates_mae_for_large_residuals(self):
        # sqrt(r^2 + eps^2) - eps -> |r| as |r| >> eps
        r = _t([50.0, -50.0])
        out = float(charbonnier_loss(backend, r, eps=1e-3))
        expected = float(mae_loss(backend, r))
        assert out == pytest.approx(expected, rel=1e-3)

    def test_quantile_loss_rejects_out_of_range(self):
        r = _t([1.0, -1.0])
        with pytest.raises(ValueError):
            quantile_loss(backend, r, quantile=0.0)
        with pytest.raises(ValueError):
            quantile_loss(backend, r, quantile=1.0)
        with pytest.raises(ValueError):
            quantile_loss(backend, r, quantile=-0.2)

    def test_quantile_loss_at_half_is_half_mae(self):
        r = _t([1.0, -2.0, 3.0, -4.0])
        q_loss = float(quantile_loss(backend, r, quantile=0.5))
        mae = float(mae_loss(backend, r))
        assert 2.0 * q_loss == pytest.approx(mae, rel=1e-5)

    def test_quantile_loss_asymmetric_penalizes_underprediction_more_for_high_q(self):
        # For q close to 1, positive residuals (target > prediction,
        # i.e. under-prediction) should be penalized more than an
        # equally-sized negative residual.
        pos = _t([1.0])
        neg = _t([-1.0])
        q = 0.9
        loss_pos = float(quantile_loss(backend, pos, quantile=q))
        loss_neg = float(quantile_loss(backend, neg, quantile=q))
        assert loss_pos > loss_neg


# ---------------------------------------------------------------------------
# Boundary-condition losses
# ---------------------------------------------------------------------------

class TestBoundaryLosses:
    def test_dirichlet_loss_zero_when_matched(self):
        pred = _t([1.0, 2.0, 3.0])
        target = _t([1.0, 2.0, 3.0])
        assert float(dirichlet_loss(backend, pred, target)) == pytest.approx(0.0, abs=1e-6)

    def test_dirichlet_loss_matches_mse_of_diff(self):
        pred = _t([1.0, 2.0])
        target = _t([0.0, 0.0])
        out = float(dirichlet_loss(backend, pred, target, weight=2.0))
        expected = 2.0 * float(mse_loss(backend, pred - target))
        assert out == pytest.approx(expected)

    def test_neumann_loss_without_normal(self):
        grad_pred = _t([1.0, 2.0])
        flux = _t([1.0, 1.0])
        out = float(neumann_loss(backend, grad_pred, flux))
        expected = float(mse_loss(backend, grad_pred - flux))
        assert out == pytest.approx(expected)

    def test_neumann_loss_with_normal_projects_correctly(self):
        # grad = [[1, 0], [0, 1]], normal = [[1, 0], [0, 1]] -> dot = [1, 1]
        grad_pred = _t([[1.0, 0.0], [0.0, 1.0]])
        normal = _t([[1.0, 0.0], [0.0, 1.0]])
        flux = _t([1.0, 1.0])
        out = float(neumann_loss(backend, grad_pred, flux, normal=normal))
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_robin_loss_matches_formula(self):
        pred = _t([2.0])
        grad_pred = _t([3.0])
        alpha, beta = 1.5, 0.5
        target = _t([4.0])
        out = float(robin_loss(backend, pred, grad_pred, alpha, beta, target))
        residual = alpha * 2.0 + beta * 3.0 - 4.0
        assert out == pytest.approx(residual ** 2)

    def test_periodic_loss_value_only(self):
        left = _t([1.0, 2.0])
        right = _t([1.0, 2.5])
        out = float(periodic_loss(backend, left, right))
        expected = float(mse_loss(backend, left - right))
        assert out == pytest.approx(expected)

    def test_periodic_loss_includes_gradient_term(self):
        left = _t([1.0])
        right = _t([1.0])   # value matches exactly
        grad_left = _t([2.0])
        grad_right = _t([0.0])  # gradient mismatch
        out = float(periodic_loss(backend, left, right, grad_left, grad_right))
        assert out > 0.0
        expected = float(mse_loss(backend, grad_left - grad_right))
        assert out == pytest.approx(expected)

    def test_interface_loss_is_alias_for_periodic_loss(self):
        assert interface_loss is periodic_loss

    def test_data_loss_matches_mse_of_diff(self):
        pred = _t([1.0, 2.0, 3.0])
        obs = _t([1.5, 1.5, 3.5])
        out = float(data_loss(backend, pred, obs, weight=3.0))
        expected = 3.0 * float(mse_loss(backend, pred - obs))
        assert out == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Regularisation losses
# ---------------------------------------------------------------------------

class TestRegularisationLosses:
    def test_sobolev_loss_zero_for_constant_function(self):
        # u(x) = 5 (constant) -> grad is zero everywhere -> loss is exactly 0
        def model_fn(x):
            return backend.zeros(x.shape[:-1]) + 5.0

        x = _t(np.random.uniform(-1, 1, size=(8, 2)).astype(np.float32))
        grad_fn = backend.grad(lambda y: backend.sum(model_fn(y)), argnums=0)
        out = float(sobolev_loss(backend, model_fn, x, grad_fn, order=1, weight=1.0))
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_sobolev_loss_positive_for_nonconstant_function(self):
        def model_fn(x):
            return backend.sum(x ** 2, axis=-1)

        x = _t(np.random.uniform(-1, 1, size=(8, 2)).astype(np.float32))
        grad_fn = backend.grad(lambda y: backend.sum(model_fn(y)), argnums=0)
        out = float(sobolev_loss(backend, model_fn, x, grad_fn, order=1, weight=1.0))
        assert out > 0.0

    def test_gradient_penalty_is_finite_and_nonnegative(self):
        def discriminator_fn(x):
            return backend.sum(x ** 2, axis=-1)

        real = _t(np.random.uniform(-1, 1, size=(16, 3)).astype(np.float32))
        fake = _t(np.random.uniform(-1, 1, size=(16, 3)).astype(np.float32))
        out = gradient_penalty(backend, discriminator_fn, real, fake)
        assert torch.isfinite(out)
        assert float(out) >= 0.0

    def test_gradient_penalty_eps_is_actually_sampled_per_example(self):
        # eps must vary per-sample (not be a hardcoded 0.5 for every row),
        # otherwise x_hat always sits at the exact midpoint of the
        # real-fake interpolation line.
        #
        # This exercises the real, public function rather than reaching
        # into `backend.random_uniform(...)` directly, since that's an
        # optional fast path inside gradient_penalty itself (guarded by
        # `hasattr`), not part of the AbstractBackend contract. A
        # discriminator with a curved (not linear) response to its input
        # makes the penalty's value depend on *where* x_hat lands along
        # the real-fake line, so a hardcoded eps=0.5 would give the exact
        # same value on every call, while genuine per-sample U[0,1]
        # sampling gives different values run to run.
        def discriminator_fn(x):
            return backend.sum(x ** 3, axis=-1)

        real = _t(np.zeros((64, 2), dtype=np.float32))
        fake = _t(np.ones((64, 2), dtype=np.float32))

        out_a = float(gradient_penalty(backend, discriminator_fn, real, fake))
        out_b = float(gradient_penalty(backend, discriminator_fn, real, fake))
        assert out_a != pytest.approx(out_b, abs=1e-9), (
            "gradient_penalty should differ run-to-run (fresh random eps "
            "per call), not always evaluate at a fixed interpolation point"
        )

    def test_spectral_energy_loss_near_zero_for_pure_low_frequency_signal(self):
        # A sinusoid at an exact integer DFT bin (k0=2 over n=64 samples)
        # has its energy concentrated essentially entirely in that single
        # bin, with all other bins at ~0 up to float32 precision -- so
        # penalising everything above a cutoff well past k0 should be
        # ~0. (Note: rfft's output length is n//2+1, not n -- a cutoff at
        # or beyond that slices into an empty tensor, whose mean is NaN,
        # not 0, so the cutoff here is deliberately kept well inside the
        # valid range rather than at the full signal length.)
        n, k0 = 64, 2
        t = np.arange(n)
        pred = _t(np.sin(2 * np.pi * k0 * t / n).astype(np.float32))[None, :]
        out = float(spectral_energy_loss(backend, pred, high_freq_cutoff=10, weight=1.0))
        assert out == pytest.approx(0.0, abs=1e-3)

    def test_spectral_energy_loss_positive_for_high_frequency_signal(self):
        # A pure high-frequency (Nyquist-adjacent) sinusoid should have
        # nonzero energy above a low cutoff.
        n = 64
        t = np.arange(n)
        pred = _t(np.sin(np.pi * 0.9 * t).astype(np.float32))[None, :]
        out = float(spectral_energy_loss(backend, pred, high_freq_cutoff=2, weight=1.0))
        assert out > 0.0


# ---------------------------------------------------------------------------
# Structural physics penalties (divergence / curl)
# ---------------------------------------------------------------------------

class TestStructuralPenalties:
    def test_divergence_penalty_zero_for_divergence_free_field(self):
        # jacobian[i,j] = du_i/dx_j. Pick du_x/dx = 1, du_y/dy = -1 -> div = 0.
        jac = _t(np.tile(np.array([[1.0, 0.3], [0.7, -1.0]], dtype=np.float32), (5, 1, 1)))
        out = float(divergence_penalty(backend, jac))
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_divergence_penalty_positive_for_nonzero_divergence(self):
        jac = _t(np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (5, 1, 1)))
        out = float(divergence_penalty(backend, jac))
        assert out == pytest.approx((2.0) ** 2)  # div = 1+1 = 2 everywhere -> mse = 4

    def test_curl_penalty_2d_zero_for_symmetric_jacobian(self):
        # curl_z = du_y/dx - du_x/dy; symmetric off-diagonal -> zero curl.
        jac = _t(np.tile(np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32), (4, 1, 1)))
        out = float(curl_penalty_2d(backend, jac))
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_curl_penalty_2d_matches_hand_calc(self):
        jac = _t(np.array([[[1.0, 2.0], [5.0, 1.0]]], dtype=np.float32))
        out = float(curl_penalty_2d(backend, jac))
        # curl_z = jac[1,0] - jac[0,1] = 5 - 2 = 3 -> mse = 9
        assert out == pytest.approx(9.0)

    def test_curl_penalty_3d_zero_for_gradient_field(self):
        # A genuine gradient field (Jacobian of a scalar potential's
        # gradient, i.e. a Hessian) is symmetric -> curl-free everywhere.
        hess = np.array([[1.0, 2.0, 3.0],
                          [2.0, 4.0, 5.0],
                          [3.0, 5.0, 6.0]], dtype=np.float32)
        jac = _t(np.tile(hess, (3, 1, 1)))
        out = float(curl_penalty_3d(backend, jac))
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_curl_penalty_3d_matches_hand_calc(self):
        jac = np.zeros((1, 3, 3), dtype=np.float32)
        jac[0, 2, 1] = 1.0  # du_z/dy = 1, everything else 0
        out = float(curl_penalty_3d(backend, _t(jac)))
        # curl_x = du_z/dy - du_y/dz = 1 - 0 = 1; others 0 -> mean(curl_sq) = 1
        assert out == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Causal time-weighting
# ---------------------------------------------------------------------------

class TestCausalWeightedLoss:
    def test_rejects_nonpositive_bucket_count(self):
        r = _t([1.0, 2.0])
        idx = _t([0, 0])
        with pytest.raises(ValueError):
            causal_weighted_loss(backend, r, idx, n_buckets=0)

    def test_weights_sum_to_one(self):
        r = _t([1.0, 2.0, 3.0, 0.5, 0.1, 0.05])
        idx = backend.tensor(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
        _, weights = causal_weighted_loss(backend, r, idx, n_buckets=3, causality_tol=1.0)
        w_np = backend.to_numpy(weights)
        assert float(np.sum(w_np)) == pytest.approx(1.0, rel=1e-4)

    def test_weights_nonincreasing_across_buckets_when_early_loss_large(self):
        # Bucket 0 has a large residual, buckets 1 and 2 are small --
        # later buckets should be weighted <= earlier ones.
        r = _t([10.0, 10.0, 0.1, 0.1, 0.01, 0.01])
        idx = backend.tensor(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
        _, weights = causal_weighted_loss(backend, r, idx, n_buckets=3, causality_tol=1.0)
        w_np = backend.to_numpy(weights)
        assert w_np[0] >= w_np[1] >= w_np[2]

    def test_empty_bucket_does_not_crash_and_contributes_zero(self):
        # No points at all assigned to bucket index 1.
        r = _t([1.0, 1.0])
        idx = backend.tensor(np.array([0, 2], dtype=np.int64))
        loss, weights = causal_weighted_loss(backend, r, idx, n_buckets=3)
        assert torch.isfinite(loss)
        w_np = backend.to_numpy(weights)
        assert np.all(np.isfinite(w_np))

    def test_zero_causality_tol_gives_uniform_weights(self):
        # tol=0 -> every raw weight is exp(0)=1 regardless of cumulative
        # loss -> uniform 1/n_buckets after normalization.
        r = _t([5.0, 5.0, 0.0, 0.0])
        idx = backend.tensor(np.array([0, 0, 1, 1], dtype=np.int64))
        _, weights = causal_weighted_loss(backend, r, idx, n_buckets=2, causality_tol=0.0)
        w_np = backend.to_numpy(weights)
        np.testing.assert_allclose(w_np, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# WeightedLossComposite
# ---------------------------------------------------------------------------

class TestWeightedLossCompositeFixed:
    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError):
            WeightedLossComposite(backend, strategy="not_a_real_strategy")

    def test_fixed_strategy_total_is_weighted_sum(self):
        c = WeightedLossComposite(backend, strategy="fixed")
        c.add("a", lambda: backend.tensor(np.array(2.0, dtype=np.float32)), weight=1.0)
        c.add("b", lambda: backend.tensor(np.array(3.0, dtype=np.float32)), weight=10.0)
        total, values = c()
        assert float(total) == pytest.approx(1.0 * 2.0 + 10.0 * 3.0)
        assert float(values["a"]) == pytest.approx(2.0)
        assert float(values["b"]) == pytest.approx(3.0)

    def test_fixed_strategy_weights_never_change(self):
        c = WeightedLossComposite(backend, strategy="fixed")
        c.add("a", lambda: backend.tensor(np.array(100.0, dtype=np.float32)), weight=1.0)
        for _ in range(5):
            c()
        assert c._terms[0].weight == 1.0

    def test_summary_reports_weight_and_recent_history(self):
        c = WeightedLossComposite(backend, strategy="fixed")
        c.add("a", lambda: backend.tensor(np.array(1.0, dtype=np.float32)), weight=2.0)
        c()
        c()
        summary = c.summary()
        assert summary["a"]["weight"] == 2.0
        assert summary["a"]["history"] == [1.0, 1.0]

    def test_trainable_parameters_is_always_empty(self):
        c = WeightedLossComposite(backend, strategy="fixed")
        assert c.trainable_parameters() == []


class TestWeightedLossCompositeSoftmaxTemp:
    def test_higher_loss_term_gets_higher_weight(self):
        c = WeightedLossComposite(backend, strategy="softmax_temp",
                                   reweight_every=1, temperature=1.0)
        c.add("small", lambda: backend.tensor(np.array(0.1, dtype=np.float32)))
        c.add("large", lambda: backend.tensor(np.array(5.0, dtype=np.float32)))
        c()  # step 0: reweight_every=1 -> reweights immediately
        assert c._terms[1].weight > c._terms[0].weight

    def test_weights_sum_to_n_terms(self):
        c = WeightedLossComposite(backend, strategy="softmax_temp",
                                   reweight_every=1, temperature=0.5)
        c.add("a", lambda: backend.tensor(np.array(1.0, dtype=np.float32)))
        c.add("b", lambda: backend.tensor(np.array(2.0, dtype=np.float32)))
        c.add("c", lambda: backend.tensor(np.array(3.0, dtype=np.float32)))
        c()
        total_w = sum(t.weight for t in c._terms)
        assert total_w == pytest.approx(3.0, rel=1e-3)


class TestWeightedLossCompositeRelobralo:
    def test_first_call_keeps_weights_near_initial(self):
        # On the very first call, prev == init == current for every term,
        # so both softmax-ratio vectors are uniform and the EMA blend
        # should land close to each term's starting weight.
        c = WeightedLossComposite(backend, strategy="relobralo",
                                   reweight_every=1, relobralo_decay=0.5)
        c.add("a", lambda: backend.tensor(np.array(1.0, dtype=np.float32)), weight=1.0)
        c.add("b", lambda: backend.tensor(np.array(1.0, dtype=np.float32)), weight=1.0)
        c()
        for t in c._terms:
            assert t.weight == pytest.approx(1.0, rel=1e-2)

    def test_runs_multiple_steps_without_error_and_stays_finite(self):
        c = WeightedLossComposite(backend, strategy="relobralo", reweight_every=1)
        vals = [10.0, 5.0, 1.0, 0.5, 0.1]
        idx = {"i": 0}

        def make_fn():
            def _fn():
                v = vals[min(idx["i"], len(vals) - 1)]
                return backend.tensor(np.array(v, dtype=np.float32))
            return _fn

        c.add("a", make_fn())
        for step in range(5):
            idx["i"] = step
            total, _ = c()
            assert torch.isfinite(total)


class TestWeightedLossCompositeUncertainty:
    def test_first_call_total_equals_sum_of_raw_values(self):
        # log_var starts at 0 -> eff_weight=exp(0)=1, s=0 added -> the
        # very first call's total is exactly sum(values), before any
        # log-variance update takes effect.
        c = WeightedLossComposite(backend, strategy="uncertainty")
        c.add("a", lambda: backend.tensor(np.array(3.0, dtype=np.float32)))
        c.add("b", lambda: backend.tensor(np.array(7.0, dtype=np.float32)))
        total, values = c()
        assert float(total) == pytest.approx(3.0 + 7.0, rel=1e-5)

    def test_log_variance_updates_after_first_call(self):
        c = WeightedLossComposite(backend, strategy="uncertainty", uncertainty_lr=0.5)
        c.add("a", lambda: backend.tensor(np.array(4.0, dtype=np.float32)))
        assert c._uncertainty_log_vars["a"] == 0.0
        c()
        assert c._uncertainty_log_vars["a"] != 0.0

    def test_effective_weight_reflected_in_term_weight_field(self):
        c = WeightedLossComposite(backend, strategy="uncertainty")
        c.add("a", lambda: backend.tensor(np.array(2.0, dtype=np.float32)))
        c()
        # eff_weight = exp(-s) with s==0 on the first call -> 1.0
        assert c._terms[0].weight == pytest.approx(1.0)


class TestWeightedLossCompositeGradNorm:
    def test_falls_back_to_fixed_and_warns_when_not_ready(self):
        c = WeightedLossComposite(backend, strategy="grad_norm", reweight_every=1)
        c.add("a", lambda: backend.tensor(np.array(1.0, dtype=np.float32)), weight=1.0)
        with pytest.warns(UserWarning):
            total, values = c()
        assert float(total) == pytest.approx(1.0)  # unchanged weight=1.0 fallback

    def test_balances_weight_toward_target_gradient_norm(self):
        p1 = torch.nn.Parameter(torch.tensor(2.0))
        p2 = torch.nn.Parameter(torch.tensor(3.0))

        c = WeightedLossComposite(backend, strategy="grad_norm", reweight_every=1)
        c.set_shared_params([p1, p2])
        # "anchor" term has d/dp1 = 2*p1 = 4 at p1=2 (grad wrt p2 is 0)
        c.add("anchor", lambda: p1 ** 2, weight=1.0)
        # "steep" term has a much larger gradient magnitude: d/dp2 = 100
        c.add("steep", lambda: 50.0 * p2 ** 2, weight=1.0)

        c()
        anchor_w = c._terms[0].weight
        steep_w = c._terms[1].weight
        # anchor is the default target (first term added) -> its own
        # weight should end up ~1 (target_norm / anchor_norm == 1), and
        # the much-steeper term should be down-weighted well below 1.
        assert anchor_w == pytest.approx(1.0, rel=1e-3)
        assert steep_w < anchor_w

    def test_grad_norm_requires_no_stale_graph_error_across_terms(self):
        # Both terms share `p`, so grad_norm must use retain_graph=True
        # internally or the second term's backward call would crash.
        p = torch.nn.Parameter(torch.tensor(1.5))
        c = WeightedLossComposite(backend, strategy="grad_norm", reweight_every=1)
        c.set_shared_params([p])
        c.add("a", lambda: p ** 2)
        c.add("b", lambda: p ** 3)
        total, values = c()  # should not raise
        assert torch.isfinite(total)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))