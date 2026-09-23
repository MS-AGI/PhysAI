"""Train a wave PINN and cross-validate it against Dedalus.

Run from the repository root with:

    python examples/wave_cross_validation.py

This example requires the optional Dedalus dependency.
"""
from __future__ import annotations

import numpy as np

from physai.backends import get_backend
from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import PDEResidual
from physai.models.pinn import build_pinn
from physai.trainer import Trainer


class FirstOrderWave(PDEResidual):
    """Wave equation written as the first-order state system u_t=v, v_t=c^2 u_xx."""

    def __init__(self, backend, c=1.0, *, diff_mode="reverse"):
        super().__init__(backend, c=c, diff_mode=diff_mode)
        self.c = c

    def __call__(self, model_fn, points):
        b = self.backend
        grad_u = lambda q: b.grad(lambda z: b.sum(model_fn(z)[..., 0]), mode=self.diff_mode)(q)
        grad_v = lambda q: b.grad(lambda z: b.sum(model_fn(z)[..., 1]), mode=self.diff_mode)(q)
        u_t = grad_u(points)[..., -1]
        v_t = grad_v(points)[..., -1]
        u_xx = b.grad(lambda q: b.sum(grad_u(q)[..., 0]), mode=self.diff_mode)(points)[..., 0]
        return b.stack([u_t - model_fn(points)[..., 1], v_t - self.c**2 * u_xx], axis=-1)


def main():
    """Train the two-state wave PINN and compare both fields with Dedalus."""
    backend = get_backend("torch", device="cpu")
    c = 1.0
    domain = DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)], time_domain=(0.0, 0.25))
    spec = ProblemSpec(
        pde_name="wave", model_arch="pinn", domain=domain, backend_name="torch",
        n_collocation=512, n_bc_points=128, max_epochs=2500, use_lbfgs_phase=False,
    )
    config = AutoOptimizer(backend).analyse(spec)
    model = build_pinn(backend, n_input=2, n_output=2, hidden_sizes=(64, 64, 64), use_fourier=True)
    residual = FirstOrderWave(backend, c=c)
    rng = np.random.default_rng(8)
    coll = np.column_stack([rng.uniform(-1, 1, 512), rng.uniform(0, 0.25, 512)]).astype(np.float32)

    x0 = rng.uniform(-1, 1, 256).astype(np.float32)
    ic_points = np.column_stack([x0, np.zeros_like(x0)]).astype(np.float32)
    ic_values = np.column_stack([np.sin(np.pi * x0), np.zeros_like(x0)]).astype(np.float32)
    bc_t = rng.uniform(0, 0.25, 128).astype(np.float32)
    bc_points = np.column_stack([np.repeat([-1.0, 1.0], 64), bc_t]).astype(np.float32)
    bc_values = np.zeros((len(bc_points), 2), dtype=np.float32)

    trainer = Trainer(
        backend, config, model, residual, collocation_points=backend.tensor(coll),
        bc_points=backend.tensor(bc_points), bc_values=backend.tensor(bc_values),
        ic_points=backend.tensor(ic_points), ic_values=backend.tensor(ic_values),
    )
    trainer.train()
    errors = trainer.cross_validate(
        domain_type="Chebyshev", bounds=(-1.0, 1.0), grid_points=128,
        variables=["u", "v"],
        equations=["dt(u) - v = 0", "dt(v) - dx_x0(dx_x0(u)) = 0"],
        bcs=["u(x0='left') = 0", "u(x0='right') = 0"],
        ics={"u": lambda x: np.sin(np.pi * x), "v": lambda x: np.zeros_like(x)},
        dt=0.002, stop_time=0.25,
    )
    print(errors)
    return trainer, errors


if __name__ == "__main__":
    main()
