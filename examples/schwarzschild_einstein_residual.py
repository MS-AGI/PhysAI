"""Fit a Schwarzschild exterior with the Einstein vacuum field residual.

Run from the repository root with:

    python examples/schwarzschild_einstein_residual.py
"""
from __future__ import annotations

import numpy as np

from physai.backends import get_backend
from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import build_residual
from physai.models.pinn import OutputScaler, build_pinn
from physai.trainer import Trainer


def schwarzschild_metric(points, schwarzschild_radius=1.0):
    """Exact isotropic-coordinate vacuum metric for coordinates [t, x, y, z]."""
    radius = np.linalg.norm(points[:, 1:4], axis=1)
    q = schwarzschild_radius / (4.0 * radius)
    lapse = (1.0 - q) / (1.0 + q)
    spatial = (1.0 + q) ** 4
    # Symmetric 4-metric order: tt, tx, ty, tz, xx, xy, xz, yy, yz, zz.
    metric = np.zeros((len(points), 10), dtype=np.float32)
    metric[:, 0] = -(lapse**2)
    metric[:, 4] = spatial
    metric[:, 7] = spatial
    metric[:, 9] = spatial
    return metric


def main():
    """Train the Einstein vacuum residual in a Schwarzschild exterior region."""
    backend = get_backend("torch", device="cpu")
    # Isotropic radius is at least sqrt(12), well outside the horizon at rs/4.
    domain = DomainSpec(spatial_dims=3, bounds=[(2.0, 3.0)] * 3, time_domain=(0.0, 0.1))
    spec = ProblemSpec(
        pde_name="einstein_field", model_arch="pinn", domain=domain,
        backend_name="torch", n_collocation=8, n_bc_points=0,
        max_epochs=25, use_lbfgs_phase=False,
    )
    config = AutoOptimizer(backend).analyse(spec)
    model = build_pinn(
        backend, n_input=4, n_output=10, hidden_sizes=(32, 32),
        output_scaler=OutputScaler(
            backend, [0.05] * 10, [-0.8, 0, 0, 0, 1.2, 0, 0, 1.2, 0, 1.2],
        ),
    )
    residual = build_residual("einstein_field", backend, diff_mode="reverse")
    rng = np.random.default_rng(9)

    def sample_spacetime(n):
        xyz = rng.uniform(2.0, 3.0, size=(n, 3))
        t = rng.uniform(0.0, 0.1, size=(n, 1))
        return np.concatenate([t, xyz], axis=1).astype(np.float32)

    coll = sample_spacetime(8)
    data_points = sample_spacetime(24)
    trainer = Trainer(
        backend, config, model, residual, collocation_points=backend.tensor(coll),
        data_points=backend.tensor(data_points),
        data_values=backend.tensor(schwarzschild_metric(data_points)),
    )
    return trainer, trainer.train()


if __name__ == "__main__":
    main()
