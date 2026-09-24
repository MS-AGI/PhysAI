"""Train a first-order wave PINN and compare it with a classical solver.

Run from the repository root with:

    python examples/wave_cross_validation.py
    python examples/wave_cross_validation.py --reference solve-box

AutoSolve is the default reference. ``solve-box`` selects the alternative
Dedalus box solver and requires the optional Dedalus dependency.
"""
from __future__ import annotations

import argparse

import numpy as np

from physai.backends import get_backend
from physai.core.auto_optimizer import AutoOptimizer, DomainSpec, ProblemSpec
from physai.core.pde_residual import PDEResidual, build_residual, register_pde
from physai.geometry import BoundaryConditionSet, box, face_region
from physai.models.pinn import build_pinn
from physai.solvers import autosolve
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


def main(reference="autosolve"):
    """Train the two-state wave PINN and compare it with one reference solve."""
    if reference not in {"autosolve", "solve-box"}:
        raise ValueError("reference must be 'autosolve' or 'solve-box'")
    backend = get_backend("torch", device="cpu")
    c = 1.0
    register_pde("first_order_wave", FirstOrderWave, meta={
        "order": 2, "nonlinear": False, "n_components": 2,
        "stiff": False, "spectral_bias": "mixed", "recommended_arch": "pinn",
    })
    domain = DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)], time_domain=(0.0, 0.25))
    spec = ProblemSpec(
        pde_name="first_order_wave", model_arch="pinn", domain=domain, backend_name="torch",
        n_collocation=512, n_bc_points=128, max_epochs=2500, use_lbfgs_phase=False,
    )
    config = AutoOptimizer(backend).analyse(spec)
    model = build_pinn(backend, n_input=2, n_output=2, hidden_sizes=(64, 64, 64), use_fourier=True)
    residual = build_residual("first_order_wave", backend, c=c)
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
    if reference == "autosolve":
        geometry = box([(-1.0, 1.0)])
        auto_bcs = BoundaryConditionSet(geometry)
        auto_bcs.add(
            "dirichlet", value=lambda x: np.zeros((len(x), 2)),
            region=face_region(axis=0, side="min"), name="left_end",
        )
        auto_bcs.add(
            "dirichlet", value=lambda x: np.zeros((len(x), 2)),
            region=face_region(axis=0, side="max"), name="right_end",
        )
        auto_solution = autosolve(
            residual, geometry,
            backend=backend,
            pde="first_order_wave",
            n_output=2,
            time_domain=(0.0, 0.25),
            boundary_conditions=auto_bcs,
            ic_points=x0[:16],
            ic_values=ic_values[:16],
            n_collocation=24,
            n_boundary=8,
            max_nfev=20,
            tolerance=1e-5,
            n_evaluation=128,
            eval_time=0.25,
            require_convergence=False,
            seed=17,
        )
        comparison_points = backend.tensor(auto_solution["coordinates"])
        reference_values = auto_solution["values"]
        names = ("u0", "u1")
        print(
            "AutoSolve diagnostics:",
            f"success={auto_solution['optimizer_result'].success}",
            f"residual_inf={auto_solution['residual_norm']:.3e}",
        )
    else:
        solve_box_errors = trainer.cross_validate(
            domain_type="Chebyshev", bounds=(-1.0, 1.0), grid_points=128,
            variables=["u", "v"],
            equations=["dt(u) - v = 0", f"dt(v) - {c**2}*dx_x0(dx_x0(u)) = 0"],
            bcs=["u(x0='left') = 0", "u(x0='right') = 0"],
            ics={"u": lambda x: np.sin(np.pi * x), "v": lambda x: np.zeros_like(x)},
            dt=0.002, stop_time=0.25,
        )
        print("Custom solve-box comparison:", solve_box_errors)
        return trainer, {"solve_box": solve_box_errors}

    prediction = np.asarray(backend.to_numpy(model.model_fn(comparison_points)))
    errors = {}
    for column, name in enumerate(names):
        reference_values_for_field = np.asarray(reference_values[name]).reshape(-1)
        difference = prediction[:, column].reshape(-1) - reference_values_for_field
        errors[f"{name}_l2_abs"] = float(np.linalg.norm(difference))
        errors[f"{name}_l2_rel"] = float(
            np.linalg.norm(difference) / (np.linalg.norm(reference_values_for_field) + 1e-12)
        )
    print("AutoSolve comparison:", errors)
    return trainer, {"autosolve": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", choices=("autosolve", "solve-box"), default="autosolve",
        help="reference solver to compare against (default: autosolve)",
    )
    main(reference=parser.parse_args().reference)
