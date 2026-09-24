"""Register the two Maxwell tensor laws from scalar components.

The LaTeX PDE parser handles scalar equations, so the covariant equations
\partial_mu F^{mu nu} = mu_0 J^nu and \partial_mu F_tilde^{mu nu} = 0
are expanded into their eight 3-D component equations before registration.

Run from the repository root with:

    python examples/maxwell_latex_register.py
"""
from __future__ import annotations

import numpy as np

from physai.backends import get_backend
from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import build_residual, register_latex_pde
from physai.geometry import BoundaryConditionSet, box, face_region
from physai.models.pinn import build_pinn
from physai.trainer import Trainer


FIELDS = ("E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "rho", "J_x", "J_y", "J_z")
COORDINATES = ("x", "y", "z", "t")

# SI Maxwell equations, with c^2 = 1/(mu_0 epsilon_0). rho and J are
# included as field components so callers can prescribe or supervise sources.
# These eight component equations are the two covariant tensor equations.
MAXWELL_EQUATIONS = (
    # nu = 0 component of partial_mu F^{mu nu} = mu_0 J^nu: Gauss's law
    r"\frac{\partial E_x}{\partial x} + \frac{\partial E_y}{\partial y} + \frac{\partial E_z}{\partial z} = \frac{rho}{eps0}",
    # Spatial components of partial_mu F^{mu nu} = mu_0 J^nu: Ampere-Maxwell law
    r"\frac{\partial E_x}{\partial t} - c^2 (\frac{\partial B_z}{\partial y} - \frac{\partial B_y}{\partial z}) + \frac{J_x}{eps0} = 0",
    r"\frac{\partial E_y}{\partial t} - c^2 (\frac{\partial B_x}{\partial z} - \frac{\partial B_z}{\partial x}) + \frac{J_y}{eps0} = 0",
    r"\frac{\partial E_z}{\partial t} - c^2 (\frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y}) + \frac{J_z}{eps0} = 0",
    # nu = 0 component of partial_mu F_tilde^{mu nu} = 0: no magnetic charge
    r"\frac{\partial B_x}{\partial x} + \frac{\partial B_y}{\partial y} + \frac{\partial B_z}{\partial z} = 0",
    # Spatial components of partial_mu F_tilde^{mu nu} = 0: Faraday's law
    r"\frac{\partial B_x}{\partial t} + \frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z} = 0",
    r"\frac{\partial B_y}{\partial t} + \frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x} = 0",
    r"\frac{\partial B_z}{\partial t} + \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y} = 0",
)


def main():
    register_latex_pde(
        "maxwell_tensor_em",
        MAXWELL_EQUATIONS,
        fields=FIELDS,
        coordinates=COORDINATES,
        parameters={"c": 1.0, "eps0": 1.0},
        meta={
            "order": 1,
            "nonlinear": False,
            "n_components": len(FIELDS),
            "stiff": False,
            "spectral_bias": "mixed",
            "recommended_arch": "pinn",
        },
    )

    # The registered definition can now be reused by any supported backend.
    backend = get_backend("torch", device="cpu")
    residual = build_residual("maxwell_tensor_em", backend, c=1.0, eps0=1.0)
    print(
        f"Registered {len(residual.equations)} Maxwell component equations "
        f"for {residual.n_components} fields on {len(residual.coordinates)} coordinates."
    )

    # Train a periodic, x-propagating vacuum plane wave in a 3-D box.
    bounds = [(-1.0, 1.0)] * 3
    time_domain = (0.0, 0.5)
    domain = DomainSpec(spatial_dims=3, bounds=bounds, time_domain=time_domain)
    spec = ProblemSpec(
        pde_name="maxwell_tensor_em",
        model_arch="pinn",
        domain=domain,
        backend_name="torch",
        n_collocation=2048,
        n_bc_points=192,
        max_epochs=3000,
        use_lbfgs_phase=False,
    )
    config = AutoOptimizer(backend).analyse(spec)
    model = build_pinn(
        backend, n_input=4, n_output=len(FIELDS),
        hidden_sizes=(96, 96, 96), use_fourier=True,
    )

    rng = np.random.default_rng(23)
    collocation = np.column_stack((
        rng.uniform(-1.0, 1.0, (2048, 3)),
        rng.uniform(*time_domain, 2048),
    )).astype(np.float32)
    initial_xyz = rng.uniform(-1.0, 1.0, (512, 3)).astype(np.float32)
    initial_points = np.column_stack((initial_xyz, np.zeros(len(initial_xyz)))).astype(np.float32)
    initial_values = np.zeros((len(initial_points), len(FIELDS)), dtype=np.float32)
    initial_values[:, FIELDS.index("E_y")] = np.sin(np.pi * initial_xyz[:, 0])
    initial_values[:, FIELDS.index("B_z")] = initial_values[:, FIELDS.index("E_y")] / 1.0

    periodic = BoundaryConditionSet(box(bounds))
    for axis in range(3):
        shift = np.zeros(3, dtype=np.float32)
        shift[axis] = bounds[axis][1] - bounds[axis][0]
        periodic.add(
            "periodic",
            region=face_region(axis=axis, side="min"),
            pair_region=face_region(axis=axis, side="max"),
            pair_map=lambda points, shift=shift: points + shift,
            name=f"periodic_axis_{axis}",
        )

    trainer = Trainer(
        backend,
        config,
        model,
        residual,
        collocation_points=backend.tensor(collocation),
        boundary_conditions=periodic,
        ic_points=backend.tensor(initial_points),
        ic_values=backend.tensor(initial_values),
    )
    trainer.train()

    # Animate E_y on the y=z=0 slice after training.
    import matplotlib.pyplot as plt
    from physai.visualization import animate_1d_solution

    x = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
    times = np.linspace(*time_domain, 50, dtype=np.float32)
    query_xyz = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    queries = np.column_stack((
        np.tile(query_xyz, (len(times), 1)),
        np.repeat(times, len(x)),
    ))
    predictions = backend.to_numpy(trainer.predict(backend.tensor(queries)))
    ey_frames = predictions[:, FIELDS.index("E_y")].reshape(len(times), len(x))
    animation = animate_1d_solution(x, ey_frames, param_values=times, ylabel="E_y")
    plt.show()
    return trainer, animation


if __name__ == "__main__":
    main()
