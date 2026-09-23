from typing import Callable, Dict
import torch
from physai.models.spectral_element import TensorCPMath, USENOCPModule, DynamicLossBalancer


class USENOPINNTrainer:
    """
    Production-grade Engine managing the training optimization process,
    dynamic NTK scale balancing, and verification passes for a
    ``USENOCPModule`` (see ``physai.models.spectral_element``).

    Tracks three loss terms, matching the reference paper (Feugmo &
    Pankaczy, arXiv:2606.02335): the PDE residual, C0 (value) continuity
    across element interfaces, and C1 (flux/derivative) continuity across
    those same interfaces — the paper is explicit that the flux penalty is
    "essential for stiff transport problems," not optional polish.
    """
    def __init__(self, model: USENOCPModule, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.balancer = DynamicLossBalancer(num_losses=3)

        # Pick the device from the model's own parameters instead of
        # hardcoding 'cuda', so this trainer also works on CPU-only setups.
        # GradScaler is only meaningful (and only enabled) for CUDA mixed
        # precision; on CPU we still create it but disabled, so
        # scaler.scale()/step()/update() below remain no-ops rather than
        # raising or silently doing nothing useful.
        self.device = next(model.parameters()).device
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=(self.amp_device_type == "cuda"))

    def train_step(self, t: torch.Tensor, nonlinear_fn: Callable) -> Dict[str, float]:
        self.optimizer.zero_grad()

        # Leverage Mixed Precision context to safe-keep VRAM and maximize tensor core execution
        with torch.amp.autocast(self.amp_device_type, enabled=(self.amp_device_type == "cuda")):
            residuals, boundaries = self.model.compute_residuals_and_boundaries(t, nonlinear_fn)

            # 1. Physics Loss evaluation across Elements
            pde_loss = torch.tensor(0.0, device=t.device, dtype=t.dtype)
            for e in range(self.model.n_elements):
                # Calculate exact L2 norm squared of the 10D residual without expansion
                pde_loss = pde_loss + TensorCPMath.compute_cp_norm_squared(residuals[e]).mean()

            # 2. Interface Grid Continuity Loss (C0 — value stitching of adjoining Elements)
            continuity_loss = torch.tensor(0.0, device=t.device, dtype=t.dtype)
            for e in range(self.model.n_elements - 1):
                # Interface condition: Right boundary of Element E must equal Left boundary of Element E+1
                r_boundary = boundaries[e]["right"]
                l_boundary = boundaries[e + 1]["left"]

                # Combine differences into a joint CP structure
                diff_factors = []
                for d in range(self.model.dims):
                    # Concatenate along the Rank axis to perform implicit addition/subtraction
                    combined = torch.cat([r_boundary[d], -1.0 * l_boundary[d]], dim=1)
                    diff_factors.append(combined)

                continuity_loss = continuity_loss + TensorCPMath.compute_cp_norm_squared(diff_factors).mean()

            # 3. Interface Flux Continuity Loss (C1 — derivative stitching of adjoining Elements)
            # Required by the reference architecture for stiff transport
            # problems: value continuity alone leaves the flux free to
            # jump across the element interface, which C0 stitching can't
            # detect or penalize.
            flux_loss = torch.tensor(0.0, device=t.device, dtype=t.dtype)
            for e in range(self.model.n_elements - 1):
                r_deriv = boundaries[e]["right_deriv"]
                l_deriv = boundaries[e + 1]["left_deriv"]

                diff_deriv_factors = []
                for d in range(self.model.dims):
                    combined = torch.cat([r_deriv[d], -1.0 * l_deriv[d]], dim=1)
                    diff_deriv_factors.append(combined)

                flux_loss = flux_loss + TensorCPMath.compute_cp_norm_squared(diff_deriv_factors).mean()

            # 4. Dynamic Optimization Gradient Scaling Routine
            # Isolate objective paths to balance stiffness differences
            shared_params = [p for p in self.model.network.parameters()]
            self.balancer.update_weights([pde_loss, continuity_loss, flux_loss], shared_params)

            total_loss = (
                (self.balancer.weights[0] * pde_loss)
                + (self.balancer.weights[1] * continuity_loss)
                + (self.balancer.weights[2] * flux_loss)
            )

        # Scale down loss, evaluate backward pass and unscale gradients
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Hard clip gradient norms to completely eliminate explosive gradient steps typical of spectral systems
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss_total": total_loss.item(),
            "loss_pde": pde_loss.item(),
            "loss_continuity": continuity_loss.item(),
            "loss_flux": flux_loss.item(),
            "weight_pde": self.balancer.weights[0].item(),
            "weight_continuity": self.balancer.weights[1].item(),
            "weight_flux": self.balancer.weights[2].item(),
        }


__all__ = ["USENOPINNTrainer", "build_spectral_element_trainer"]


from typing import List, Optional, Tuple, TYPE_CHECKING
from physai.models.spectral_element import build_separable_linear_operator

if TYPE_CHECKING:
    from physai.core.auto_optimizer import RuntimeConfig


def build_spectral_element_trainer(
    config: "RuntimeConfig",
    dim_coeffs: List[Dict[int, float]],
    element_widths: Optional[torch.Tensor] = None,
    lr: float = 1e-4,
) -> Tuple[USENOCPModule, USENOPINNTrainer]:
    """
    Build a ready-to-train ``(USENOCPModule, USENOPINNTrainer)`` pair from
    a ``RuntimeConfig`` whose ``config.model.arch == "spectral_element"``
    — the same role ``build_pinn``/``build_fno`` play for the "pinn"/"fno"
    archs: pairing an ``AutoOptimizer``-sized architecture with the object
    that actually trains it. This is the integration point between
    ``AutoOptimizer``'s heuristics and the USENO spectral-element model —
    call this instead of constructing ``USENOCPModule``/``USENOPINNTrainer``
    by hand once you have a ``RuntimeConfig``.

    ``AutoOptimizer`` can size ``n_elements``/``n_modes``/``rank`` from the
    problem's stiffness/component-count heuristics (see
    ``physai.core.auto_optimizer._build_model_config``), but it cannot
    know the PDE's actual linear operator — that remains genuinely
    problem-specific information this factory still needs from the
    caller, via ``dim_coeffs`` (see ``build_separable_linear_operator``'s
    docstring for its exact format: a per-axis dict of derivative-order ->
    coefficient). This is disclosed rather than hidden behind a fake
    "fully automatic" API: AutoOptimizer sizes the network, the physics
    still has to come from you — the same way ``build_residual(pde_name,
    backend, **params)`` still needs its own PDE-specific ``params`` for
    the "pinn"/"fno" archs elsewhere in the library.

    Note ``config.problem.domain.time_domain`` must not be ``None`` —
    ``USENOCPModule``'s backbone network takes a single scalar time input
    (spatial structure is represented via the CP-factored Chebyshev
    coefficients, not the network's input), so this arch only makes sense
    for time-dependent problems, and ``AutoOptimizer._select_model_arch``
    only ever selects it in that case.
    """
    if config.model.arch != "spectral_element":
        raise ValueError(
            "build_spectral_element_trainer requires config.model.arch == "
            f"'spectral_element', got '{config.model.arch}'."
        )
    if config.problem.domain.time_domain is None:
        raise ValueError(
            "build_spectral_element_trainer: USENOCPModule's backbone "
            "network takes a scalar time input, so config.problem.domain."
            "time_domain must be set (this arch is for time-dependent "
            "problems only)."
        )

    n_elements = config.model.n_elements
    n_modes = config.model.n_modes
    rank = config.model.rank
    if n_elements is None or n_modes is None or rank is None:
        raise ValueError(
            "build_spectral_element_trainer: config.model.n_elements/"
            "n_modes/rank must all be set — they should have been filled "
            "in by AutoOptimizer._build_model_config for arch="
            "'spectral_element'; got a ModelConfig missing one of them."
        )
    dims = config.problem.domain.spatial_dims

    element_operators = build_separable_linear_operator(
        n_elements, dims, n_modes, dim_coeffs, element_widths,
    )
    model = USENOCPModule(n_elements, dims, n_modes, rank, element_operators, element_widths)
    usenopinn = USENOPINNTrainer(model, lr=lr)
    return model, usenopinn #you see- incomplete!