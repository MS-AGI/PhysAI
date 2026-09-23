"""Train a user-registered 1-D vacuum Maxwell PINN and animate its Ey field.

Run from the repository root with:

    python examples/maxwell_animation.py
"""
from __future__ import annotations

import numpy as np

from physai.backends import get_backend
from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import PDEResidual, build_residual, register_pde
from physai.geometry import BoundaryConditionSet, box, face_region
from physai.models.pinn import build_pinn
from physai.trainer import Trainer


class Maxwell1DVacuum(PDEResidual):
    """One-dimensional vacuum Maxwell system with fields [Ex, Ey, Ez, Bx, By, Bz]."""

    def __init__(self, backend, c=1.0, *, diff_mode="reverse"):
        super().__init__(backend, c=c, diff_mode=diff_mode)
        self.c = c

    def __call__(self, model_fn, points):
        b = self.backend
        grads = [
            b.grad(lambda q, j=j: b.sum(model_fn(q)[..., j]), mode=self.diff_mode)(points)
            for j in range(6)
        ]
        ex_x, ex_t = grads[0][..., 0], grads[0][..., -1]
        ey_x, ey_t = grads[1][..., 0], grads[1][..., -1]
        ez_x, ez_t = grads[2][..., 0], grads[2][..., -1]
        bx_x, bx_t = grads[3][..., 0], grads[3][..., -1]
        by_x, by_t = grads[4][..., 0], grads[4][..., -1]
        bz_x, bz_t = grads[5][..., 0], grads[5][..., -1]
        # Faraday's law, Ampere's law in vacuum, and the 1-D divergence constraints.
        return b.stack([
            ex_t, ey_t + self.c**2 * bz_x, ez_t - self.c**2 * by_x,
            bx_t, by_t - ez_x, bz_t + ey_x, ex_x, bx_x,
        ], axis=-1)


def main():
    """Register, train, and animate a periodic transverse Maxwell wave."""
    import matplotlib.pyplot as plt
    from physai.visualization import animate_1d_solution

    c = 1.0
    register_pde("maxwell_1d_vacuum", Maxwell1DVacuum, meta={
        "order": 1, "nonlinear": False, "n_components": 6,
        "stiff": False, "spectral_bias": "mixed", "recommended_arch": "pinn",
    })

    backend = get_backend("torch", device="cpu")
    domain = DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)], time_domain=(0.0, 0.5))
    spec = ProblemSpec(
        pde_name="maxwell_1d_vacuum", domain=domain, backend_name="torch",
        n_collocation=512, n_bc_points=128, max_epochs=3000,
        use_lbfgs_phase=False,
    )
    config = AutoOptimizer(backend).analyse(spec)
    model = build_pinn(backend, n_input=2, n_output=6, hidden_sizes=(64, 64, 64), use_fourier=True)
    residual = build_residual("maxwell_1d_vacuum", backend, c=c)

    rng = np.random.default_rng(7)
    coll = np.column_stack([rng.uniform(-1, 1, 512), rng.uniform(0, 0.5, 512)]).astype(np.float32)
    initial_x = rng.uniform(-1, 1, 256).astype(np.float32)
    ic_points = np.column_stack([initial_x, np.zeros_like(initial_x)]).astype(np.float32)
    ic_values = np.zeros((len(initial_x), 6), dtype=np.float32)
    ic_values[:, 1] = np.sin(np.pi * initial_x)
    ic_values[:, 5] = ic_values[:, 1] / c

    periodic = BoundaryConditionSet(box(domain.bounds))
    periodic.add(
        "periodic", region=face_region(axis=0, side="min"),
        pair_region=face_region(axis=0, side="max"),
        pair_map=lambda x: x + np.array([2.0]), name="periodic_x",
    )
    trainer = Trainer(
        backend, config, model, residual, collocation_points=backend.tensor(coll),
        boundary_conditions=periodic, ic_points=backend.tensor(ic_points),
        ic_values=backend.tensor(ic_values),
    )
    trainer.train()

    x = np.linspace(-1, 1, 200, dtype=np.float32)
    times = np.linspace(0, 0.5, 50, dtype=np.float32)
    queries = np.column_stack([np.tile(x, len(times)), np.repeat(times, len(x))])
    fields = backend.to_numpy(trainer.predict(backend.tensor(queries)))
    frames = fields[:, 1].reshape(len(times), len(x))
    animation = animate_1d_solution(x, frames, param_values=times, ylabel="Ey")
    plt.show()
    return trainer, animation


if __name__ == "__main__":
    main()
