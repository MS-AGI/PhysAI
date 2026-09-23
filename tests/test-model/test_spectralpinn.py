"""
tests/test_spectralpinn.py

Tests for physai.models.spectral_element (TensorCPMath, DynamicLossBalancer,
Chebyshev analysis/synthesis helpers, build_separable_linear_operator,
USENOCPModule) and physai.models.spectralpinn (USENOPINNTrainer,
build_spectral_element_trainer). Torch-only (this module is pure PyTorch,
unlike the other backend-agnostic models).
"""
import numpy as np
import pytest
import torch

from physai.models.spectral_element import (
    TensorCPMath,
    DynamicLossBalancer,
    USENOCPModule,
    build_separable_linear_operator,
    _chebyshev_gauss_lobatto_nodes,
    _chebyshev_vandermonde,
    _chebyshev_deriv_boundary_vectors,
    _chebyshev_coefficient_derivative_matrix,
)
from physai.models.spectralpinn import USENOPINNTrainer, build_spectral_element_trainer
from physai.core.auto_optimizer import (
    RuntimeConfig, ProblemSpec, DomainSpec, ModelConfig, TrainingConfig,
    SchedulerConfig, PDEMeta,
)


# ---------------------------------------------------------------------------
# TensorCPMath
# ---------------------------------------------------------------------------

class TestTensorCPMath:
    def test_rank1_inner_product_matches_elementwise_dot(self):
        """For rank-1 CP factors, <A,B> collapses to a product-of-dot-
        products over dims, which for a single dim is just a plain dot."""
        A = [torch.tensor([[[1.0, 2.0, 3.0]]])]  # (Batch=1, Rank=1, N=3)
        B = [torch.tensor([[[4.0, 5.0, 6.0]]])]
        result = TensorCPMath.compute_cp_inner_product(A, B)
        expected = 1 * 4 + 2 * 5 + 3 * 6
        np.testing.assert_allclose(result.numpy(), [expected])

    def test_two_dims_multiply_across_dims(self):
        """<A,B> = sum_r1,r2 prod_d <A_d[r1], B_d[r2]> — for rank 1, this
        is just the product of the two per-dimension dot products."""
        A = [
            torch.tensor([[[1.0, 0.0]]]),   # dim 0
            torch.tensor([[[0.0, 1.0]]]),   # dim 1
        ]
        B = [
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[0.0, 1.0]]]),
        ]
        result = TensorCPMath.compute_cp_inner_product(A, B)
        # dim0 dot = 1, dim1 dot = 1 -> product = 1
        np.testing.assert_allclose(result.numpy(), [1.0])

    def test_norm_squared_is_inner_product_with_self(self):
        A = [torch.tensor([[[2.0, 0.0, 0.0]]])]
        norm_sq = TensorCPMath.compute_cp_norm_squared(A)
        inner = TensorCPMath.compute_cp_inner_product(A, A)
        np.testing.assert_allclose(norm_sq.numpy(), inner.numpy())
        np.testing.assert_allclose(norm_sq.numpy(), [4.0])

    def test_complex_inner_product_uses_conjugate_and_returns_real(self):
        A = [torch.tensor([[[1.0 + 1.0j]]])]
        B = [torch.tensor([[[1.0 + 1.0j]]])]
        result = TensorCPMath.compute_cp_inner_product(A, B)
        # <a, a> with conjugation = |a|^2 = 2, real-valued
        assert not result.is_complex()
        np.testing.assert_allclose(result.numpy(), [2.0], atol=1e-6)

    def test_batch_dimension_is_independent(self):
        A = [torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])]  # Batch=2, Rank=1, N=2
        B = [torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])]
        result = TensorCPMath.compute_cp_inner_product(A, B)
        np.testing.assert_allclose(result.numpy(), [1.0, 0.0])


# ---------------------------------------------------------------------------
# DynamicLossBalancer
# ---------------------------------------------------------------------------

class TestDynamicLossBalancer:
    def test_initial_weights_are_ones(self):
        balancer = DynamicLossBalancer(num_losses=3)
        np.testing.assert_allclose(balancer.weights.numpy(), [1.0, 1.0, 1.0])

    def test_update_weights_normalises_to_num_losses(self):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(4))
        losses = [
            (param ** 2).sum(),
            (2.0 * param ** 2).sum(),
            (0.5 * param).sum(),
        ]
        balancer = DynamicLossBalancer(num_losses=3, alpha=0.0)  # alpha=0 -> pure target weights
        balancer.update_weights(losses, [param])
        assert torch.isfinite(balancer.weights).all()
        np.testing.assert_allclose(balancer.weights.sum().item(), 3.0, atol=1e-4)

    def test_update_weights_handles_unused_parameter(self):
        """Both losses have a real grad_fn (depend on `used`), but neither
        depends on `unused` at all — torch.autograd.grad's allow_unused=True
        path must return None for `unused` without crashing, and
        update_weights must still produce finite weights."""
        used = torch.nn.Parameter(torch.randn(3))
        unused = torch.nn.Parameter(torch.randn(3))
        losses = [(used ** 2).sum(), (used ** 3).sum()]
        balancer = DynamicLossBalancer(num_losses=2)
        balancer.update_weights(losses, [used, unused])
        assert torch.isfinite(balancer.weights).all()

    def test_update_weights_handles_complex_loss(self):
        param = torch.nn.Parameter(torch.randn(2))
        complex_loss = (param ** 2).sum().to(torch.complex64)
        real_loss = (param ** 2).sum()
        balancer = DynamicLossBalancer(num_losses=2)
        balancer.update_weights([complex_loss, real_loss], [param])
        assert torch.isfinite(balancer.weights).all()


# ---------------------------------------------------------------------------
# Chebyshev analysis / synthesis helpers
# ---------------------------------------------------------------------------

class TestChebyshevHelpers:
    def test_gauss_lobatto_endpoints_are_plus_minus_one(self):
        nodes = _chebyshev_gauss_lobatto_nodes(9)
        np.testing.assert_allclose(nodes[[0, -1]], [1.0, -1.0])

    def test_vandermonde_matches_known_chebyshev_polynomials(self):
        x = np.linspace(-1.0, 1.0, 11, dtype=np.float64)
        V = _chebyshev_vandermonde(x, n_modes=4)
        # T_0=1, T_1=x, T_2=2x^2-1, T_3=4x^3-3x
        np.testing.assert_allclose(V[:, 0], np.ones(11), atol=1e-10)
        np.testing.assert_allclose(V[:, 1], x, atol=1e-10)
        np.testing.assert_allclose(V[:, 2], 2 * x ** 2 - 1, atol=1e-10)
        np.testing.assert_allclose(V[:, 3], 4 * x ** 3 - 3 * x, atol=1e-10)

    def test_deriv_boundary_vectors_match_analytic_formula(self):
        n_modes = 5
        left, right = _chebyshev_deriv_boundary_vectors(n_modes)
        i = np.arange(n_modes)
        expected_right = i ** 2
        expected_left = ((-1.0) ** (i + 1)) * (i ** 2)
        np.testing.assert_allclose(right, expected_right)
        np.testing.assert_allclose(left, expected_left)

    def test_coefficient_derivative_matrix_matches_numpy_chebder(self):
        """Verified the same way the module's own docstring says it was
        verified: against numpy.polynomial.chebyshev.chebder for every
        basis vector T_0..T_{n-1}."""
        n_modes = 6
        D = _chebyshev_coefficient_derivative_matrix(n_modes, dtype=np.float64)
        for k in range(n_modes):
            coeffs = np.zeros(n_modes)
            coeffs[k] = 1.0
            expected = np.polynomial.chebyshev.chebder(coeffs, m=1)
            got = D[:, k][: len(expected)]
            np.testing.assert_allclose(got, expected, atol=1e-10)
            # Anything beyond len(expected) must be zero (chebder drops the
            # trailing zero coefficient that would appear past the reduced degree).
            np.testing.assert_allclose(D[:, k][len(expected):], 0.0, atol=1e-10)

    def test_derivative_of_x_is_one(self):
        """D @ e_1 == e_0  (d/dx[x] = 1), as the docstring claims."""
        D = _chebyshev_coefficient_derivative_matrix(4, dtype=np.float64)
        e1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        result = D @ e1
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_derivative_of_T2_is_4T1(self):
        """D @ e_2 == 4*e_1  (d/dx[2x^2-1] = 4x), as the docstring claims."""
        D = _chebyshev_coefficient_derivative_matrix(4, dtype=np.float64)
        e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        result = D @ e2
        np.testing.assert_allclose(result, [0.0, 4.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# build_separable_linear_operator
# ---------------------------------------------------------------------------

class TestBuildSeparableLinearOperator:
    def test_output_shape(self):
        ops = build_separable_linear_operator(
            n_elements=2, dims=2, n_modes=5, dim_coeffs=[{2: 1.0}, {2: 1.0}],
        )
        assert ops.shape == (2, 2, 5, 5)

    def test_dim_coeffs_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_separable_linear_operator(
                n_elements=1, dims=2, n_modes=4, dim_coeffs=[{2: 1.0}],
            )

    def test_element_widths_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_separable_linear_operator(
                n_elements=2, dims=1, n_modes=4, dim_coeffs=[{2: 1.0}],
                element_widths=torch.ones(3),
            )

    def test_first_derivative_operator_matches_coefficient_derivative_matrix(self):
        """dim_coeffs={1: 1.0} with element_widths=2.0 (so the chain-rule
        factor 2/width == 1) must reproduce the raw Chebyshev
        coefficient-space derivative matrix exactly."""
        n_modes = 5
        ops = build_separable_linear_operator(
            n_elements=1, dims=1, n_modes=n_modes, dim_coeffs=[{1: 1.0}],
            element_widths=torch.tensor([2.0]),
        )
        D = _chebyshev_coefficient_derivative_matrix(n_modes, dtype=np.float32)
        np.testing.assert_allclose(ops[0, 0], D, atol=1e-5)

    def test_default_width_applies_chain_rule_factor_of_two(self):
        """The default element_widths=1 still applies the reference-domain
        chain-rule factor 2/width = 2 (the reference domain [-1,1] has
        width 2, so a physical width of 1 is NOT the identity mapping)."""
        n_modes = 5
        ops = build_separable_linear_operator(
            n_elements=1, dims=1, n_modes=n_modes, dim_coeffs=[{1: 1.0}],
        )
        D = _chebyshev_coefficient_derivative_matrix(n_modes, dtype=np.float32)
        np.testing.assert_allclose(ops[0, 0], 2.0 * D, atol=1e-5)

    def test_width_scaling_applies_chain_rule(self):
        """Halving the element width doubles a first-derivative operator
        (d/dx_phys = (2/width) * d/dxi_ref)."""
        n_modes = 4
        ops_w1 = build_separable_linear_operator(
            n_elements=1, dims=1, n_modes=n_modes, dim_coeffs=[{1: 1.0}],
            element_widths=torch.tensor([1.0]),
        )
        ops_w_half = build_separable_linear_operator(
            n_elements=1, dims=1, n_modes=n_modes, dim_coeffs=[{1: 1.0}],
            element_widths=torch.tensor([0.5]),
        )
        np.testing.assert_allclose(ops_w_half[0, 0], 2.0 * ops_w1[0, 0], atol=1e-5)

    def test_zeroth_order_reaction_term_is_identity_scaled(self):
        n_modes = 3
        ops = build_separable_linear_operator(
            n_elements=1, dims=1, n_modes=n_modes, dim_coeffs=[{0: 3.0}],
        )
        np.testing.assert_allclose(ops[0, 0], 3.0 * np.eye(n_modes), atol=1e-5)

    def test_invalid_negative_order_raises(self):
        with pytest.raises(ValueError):
            build_separable_linear_operator(
                n_elements=1, dims=1, n_modes=4, dim_coeffs=[{-1: 1.0}],
            )


# ---------------------------------------------------------------------------
# USENOCPModule
# ---------------------------------------------------------------------------

def _make_module(n_elements=2, dims=1, n_modes=4, rank=2):
    ops = build_separable_linear_operator(
        n_elements, dims, n_modes, dim_coeffs=[{2: 1.0}] * dims,
    )
    return USENOCPModule(n_elements, dims, n_modes, rank, ops)


class TestUSENOCPModule:
    def test_boundary_vectors_match_chebyshev_endpoints(self):
        module = _make_module(n_modes=5)
        i = np.arange(5)
        np.testing.assert_allclose(module.left_boundary_vector.numpy(), (-1.0) ** i)
        np.testing.assert_allclose(module.right_boundary_vector.numpy(), np.ones(5))

    def test_get_cp_factors_shape(self):
        module = _make_module(n_elements=3, dims=2, n_modes=4, rank=2)
        t = torch.rand(5, 1)
        factors = module.get_cp_factors(t)
        assert len(factors) == 3  # n_elements
        assert len(factors[0]) == 2  # dims
        assert factors[0][0].shape == (5, 2, 4)  # (Batch, Rank, N_modes)

    def test_pseudospectral_product_of_constants_is_constant(self):
        """f(x)=1 (Chebyshev coeff vector [1,0,0,...]) times itself must
        equal 1 everywhere -> the same coefficient vector back out."""
        module = _make_module(n_modes=4)
        const_coeffs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (Batch=1, N_modes)
        result = module.pseudospectral_1d_product(const_coeffs, const_coeffs)
        np.testing.assert_allclose(result.numpy(), const_coeffs.numpy(), atol=1e-4)

    def test_pseudospectral_product_with_custom_nonlinear_fn(self):
        """Passing nonlinear_fn=lambda a,b: a+b instead of the default
        product must change the result (sum of two constants, not product)."""
        module = _make_module(n_modes=4)
        a = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        b = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
        result = module.pseudospectral_1d_product(a, b, nonlinear_fn=lambda x, y: x + y)
        # constant 2 + constant 3 = constant 5 -> coeff vector [5,0,0,0]
        np.testing.assert_allclose(result.numpy(), [[5.0, 0.0, 0.0, 0.0]], atol=1e-4)

    def test_compute_residuals_and_boundaries_shapes(self):
        module = _make_module(n_elements=3, dims=1, n_modes=4, rank=2)
        t = torch.rand(2, 1)
        residuals, boundaries = module.compute_residuals_and_boundaries(t, nonlinear_fn=None)
        assert len(residuals) == 3
        assert len(boundaries) == 3
        for e_res in residuals:
            assert len(e_res) == 1  # dims
            # rank*dims (linear) + rank^2 (nonlinear) = 2 + 4 = 6 total factors
            assert e_res[0].shape[1] == 2 + 2 * 2

    def test_boundary_keys_present(self):
        module = _make_module(n_elements=2, dims=1, n_modes=4, rank=1)
        t = torch.rand(1, 1)
        _, boundaries = module.compute_residuals_and_boundaries(t, nonlinear_fn=None)
        for b in boundaries:
            assert set(b.keys()) == {"left", "right", "left_deriv", "right_deriv"}

    def test_module_is_nn_module_with_registered_buffers(self):
        module = _make_module()
        assert isinstance(module, torch.nn.Module)
        buf_names = dict(module.named_buffers()).keys()
        assert "linear_operators" in buf_names
        assert "left_boundary_vector" in buf_names
        assert "cheb_synth" in buf_names

    def test_parameters_come_from_backbone_network_only(self):
        module = _make_module()
        params = list(module.parameters())
        network_params = list(module.network.parameters())
        assert len(params) == len(network_params)

    def test_evaluate_reconstructs_constant_field_across_elements(self):
        module = _make_module(n_elements=2, dims=1, n_modes=4, rank=1)
        times = torch.tensor([[0.0], [0.5], [1.0]])
        coords = torch.tensor([[-1.0], [0.0], [1.0]])
        constant_factors = [
            [torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 3)],
            [torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 3)],
        ]
        module.get_cp_factors = lambda _t: constant_factors
        result = module.evaluate(times, coords, [(-1.0, 1.0)])
        torch.testing.assert_close(result, torch.ones((3, 1)))

    def test_model_parameters_keep_torch_module_iterator_contract(self):
        module = _make_module()
        assert isinstance(next(module.parameters()), torch.nn.Parameter)
        assert next(module.named_parameters())[0].startswith("network.")


# ---------------------------------------------------------------------------
# USENOPINNTrainer
# ---------------------------------------------------------------------------

class TestUSENOPINNTrainer:
    def test_train_step_returns_expected_keys_and_finite_losses(self):
        module = _make_module(n_elements=2, dims=1, n_modes=4, rank=2)
        trainer = USENOPINNTrainer(module, lr=1e-3)
        t = torch.rand(3, 1)
        metrics = trainer.train_step(t, nonlinear_fn=None)

        expected_keys = {
            "loss_total", "loss_pde", "loss_continuity", "loss_flux",
            "weight_pde", "weight_continuity", "weight_flux",
        }
        assert set(metrics.keys()) == expected_keys
        for v in metrics.values():
            assert np.isfinite(v)

    def test_train_step_updates_parameters(self):
        module = _make_module(n_elements=2, dims=1, n_modes=4, rank=2)
        trainer = USENOPINNTrainer(module, lr=1e-2)
        before = [p.clone() for p in module.parameters()]
        t = torch.rand(4, 1)
        trainer.train_step(t, nonlinear_fn=None)
        after = list(module.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_device_selected_from_model_parameters(self):
        module = _make_module()
        trainer = USENOPINNTrainer(module)
        assert trainer.device == next(module.parameters()).device
        assert trainer.amp_device_type == "cpu"
        assert trainer.scaler.is_enabled() is False


# ---------------------------------------------------------------------------
# build_spectral_element_trainer
# ---------------------------------------------------------------------------

def _make_config(arch="spectral_element", time_domain=(0.0, 1.0),
                  n_elements=2, n_modes=4, rank=2, spatial_dims=1):
    domain = DomainSpec(spatial_dims=spatial_dims, bounds=[(-1.0, 1.0)] * spatial_dims,
                         time_domain=time_domain)
    problem = ProblemSpec(pde_name="heat", domain=domain)
    model = ModelConfig(
        arch=arch, layer_sizes=(1, 1), activation="tanh", use_residual=False,
        n_elements=n_elements, n_modes=n_modes, rank=rank,
    )
    training = TrainingConfig(
        optimizer_name="adam", learning_rate=1e-3, weight_decay=0.0, batch_size=32,
        n_collocation=100, n_bc_points=20, loss_weights={}, scheduler=SchedulerConfig("none"),
        use_lbfgs_phase=False, lbfgs_max_iter=50, rar_enabled=False, rar_interval=1000,
        rar_fraction=0.1, dtype="float32", grad_clip_norm=None, use_jit=False,
    )
    meta = PDEMeta("heat", 2, False, 1, True, "low", "pinn")
    return RuntimeConfig(problem=problem, model=model, training=training, meta=meta)


class TestBuildSpectralElementTrainer:
    def test_wrong_arch_raises(self):
        config = _make_config(arch="pinn")
        with pytest.raises(ValueError, match="spectral_element"):
            build_spectral_element_trainer(config, dim_coeffs=[{2: 1.0}])

    def test_missing_time_domain_raises(self):
        config = _make_config(time_domain=None)
        with pytest.raises(ValueError, match="time_domain"):
            build_spectral_element_trainer(config, dim_coeffs=[{2: 1.0}])

    def test_missing_model_sizing_raises(self):
        config = _make_config(n_elements=None)
        with pytest.raises(ValueError, match="n_elements"):
            build_spectral_element_trainer(config, dim_coeffs=[{2: 1.0}])

    def test_successful_build_returns_module_and_trainer(self):
        config = _make_config(n_elements=2, n_modes=4, rank=2, spatial_dims=1)
        model, trainer = build_spectral_element_trainer(config, dim_coeffs=[{2: 1.0}])
        assert isinstance(model, USENOCPModule)
        assert isinstance(trainer, USENOPINNTrainer)
        assert model.n_elements == 2
        assert model.n_modes == 4
        assert model.rank == 2
        assert model.dims == 1

    def test_end_to_end_train_step_runs(self):
        config = _make_config(n_elements=2, n_modes=4, rank=2, spatial_dims=1)
        model, trainer = build_spectral_element_trainer(config, dim_coeffs=[{2: 1.0}])
        t = torch.rand(3, 1)
        metrics = trainer.train_step(t, nonlinear_fn=None)
        assert np.isfinite(metrics["loss_total"])
