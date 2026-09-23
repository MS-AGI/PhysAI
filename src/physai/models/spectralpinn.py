"""Backend-neutral training helpers for the USENO spectral-element PINN."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from physai.backends import AbstractBackend, get_backend
from physai.models.spectral_element import (
    TensorCPMath, USENOCPModule, DynamicLossBalancer, build_separable_linear_operator,
)

if TYPE_CHECKING:
    from physai.core.auto_optimizer import RuntimeConfig


class USENOPINNTrainer:
    """Optimizes a spectral-element model on any registered backend."""

    def __init__(self, model: USENOCPModule, lr: float = 1e-4):
        self.model = model
        self.backend: AbstractBackend = model.backend
        # Gradient inspection in DynamicLossBalancer uses torch.autograd.
        # The other backends keep the same three objectives at unit weight
        # and differentiate them through their native training APIs.
        self.balancer = DynamicLossBalancer(num_losses=3) if self.backend.name == "torch" else None
        model_parameters = list(model.parameters())
        self.device = (
            model_parameters[0].device
            if self.backend.name == "torch" and model_parameters
            else self.backend.default_device()
        )
        if self.backend.name == "torch":
            import torch
            self.amp_device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.amp_device_type == "cuda")
        self.optimizer = self.backend.build_optimizer(
            model.network, "adamw", lr=lr, weight_decay=1e-4,
        )
        self._jax_opt_state = None
        self._jax_params = model.init_jax_params(self.backend.zeros((1, 1))) if self.backend.name == "jax" else None
        if self.backend.name == "jax":
            self._jax_opt_state = self.optimizer.init(self._jax_params)

    def _losses(self, t, nonlinear_fn):
        b = self.backend
        residuals, boundaries = self.model.compute_residuals_and_boundaries(t, nonlinear_fn)
        pde_loss = b.sum(b.zeros((1,)))
        continuity_loss = b.sum(b.zeros((1,)))
        flux_loss = b.sum(b.zeros((1,)))
        for residual in residuals:
            pde_loss = pde_loss + b.mean(TensorCPMath.compute_cp_norm_squared(residual, backend=b))
        for e in range(self.model.n_elements - 1):
            value_diff = [b.concatenate([boundaries[e]["right"][d], -boundaries[e + 1]["left"][d]], axis=1)
                          for d in range(self.model.dims)]
            deriv_diff = [b.concatenate([boundaries[e]["right_deriv"][d], -boundaries[e + 1]["left_deriv"][d]], axis=1)
                           for d in range(self.model.dims)]
            continuity_loss = continuity_loss + b.mean(TensorCPMath.compute_cp_norm_squared(value_diff, backend=b))
            flux_loss = flux_loss + b.mean(TensorCPMath.compute_cp_norm_squared(deriv_diff, backend=b))
        total = pde_loss + continuity_loss + flux_loss
        return total, pde_loss, continuity_loss, flux_loss

    def _metrics(self, losses):
        vals = [float(self.backend.to_numpy(v)) for v in losses]
        return dict(zip(("loss_total", "loss_pde", "loss_continuity", "loss_flux"), vals)) | {
            "weight_pde": 1.0, "weight_continuity": 1.0, "weight_flux": 1.0,
        }

    def train_step(self, t, nonlinear_fn: Optional[Callable]):
        b = self.backend
        if b.name == "jax":
            import jax
            import optax
            params = self._jax_params
            def loss_with_aux(p):
                self.model.set_jax_params(p)
                losses = self._losses(t, nonlinear_fn)
                return losses[0], losses[1:]
            (total, breakdown), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(params)
            losses = (total, *breakdown)
            updates, self._jax_opt_state = self.optimizer.update(grads, self._jax_opt_state, params)
            self._jax_params = optax.apply_updates(params, updates)
            self.model.set_jax_params(self._jax_params)
            return self._metrics(losses)

        if b.name == "tensorflow":
            import tensorflow as tf
            with tf.GradientTape() as tape:
                losses = self._losses(t, nonlinear_fn)
            b.optimizer_step(self.optimizer, losses[0], model=self.model.network, tape=tape)
        else:
            b.zero_grad(self.model.network)
            losses = self._losses(t, nonlinear_fn)
            if b.name == "torch":
                params = list(self.model.parameters())
                assert self.balancer is not None
                self.balancer.update_weights(list(losses[1:]), params)
                total = sum(self.balancer.weights[i] * losses[i + 1] for i in range(3))
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                import torch
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                losses = (total, *losses[1:])
            else:
                b.optimizer_step(self.optimizer, losses[0], model=self.model.network)
        metrics = self._metrics(losses)
        if b.name == "torch":
            metrics.update({"weight_pde": float(self.balancer.weights[0]),
                            "weight_continuity": float(self.balancer.weights[1]),
                            "weight_flux": float(self.balancer.weights[2])})
        return metrics


def build_spectral_element_trainer(
    config: "RuntimeConfig",
    dim_coeffs: List[Dict[int, float]],
    element_widths=None,
    lr: float = 1e-4,
    backend: Optional[AbstractBackend] = None,
) -> Tuple[USENOCPModule, USENOPINNTrainer]:
    """Build a backend-native ``(model, trainer)`` pair for spectral elements."""
    if config.model.arch != "spectral_element":
        raise ValueError("build_spectral_element_trainer requires config.model.arch == 'spectral_element'.")
    if config.problem.domain.time_domain is None:
        raise ValueError("USENOCPModule requires config.problem.domain.time_domain to be set.")
    n_elements, n_modes, rank = config.model.n_elements, config.model.n_modes, config.model.rank
    if n_elements is None or n_modes is None or rank is None:
        raise ValueError("config.model.n_elements/n_modes/rank must be set for spectral_element.")
    b = backend or get_backend(config.problem.backend_name)
    dims = config.problem.domain.spatial_dims
    if element_widths is None:
        x_lo, x_hi = config.problem.domain.bounds[0]
        element_widths = [(x_hi - x_lo) / n_elements] * n_elements
    operators = build_separable_linear_operator(n_elements, dims, n_modes, dim_coeffs, element_widths)
    model = USENOCPModule(n_elements, dims, n_modes, rank, operators, element_widths, backend=b)
    return model, USENOPINNTrainer(model, lr=lr)


__all__ = ["USENOPINNTrainer", "build_spectral_element_trainer"]
