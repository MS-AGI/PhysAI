# PhysAI: A Multi-Backend Physics-Informed Neural Network Library for PDE Solving at Research Scale


[![PyPI version](https://img.shields.io/pypi/v/physai.svg)](https://pypi.org/project/physai/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/AGPL_License_3.0-indigo.svg)](https://opensource.org/licenses/agpl-3-0)
![PyPI - Total Downloads](https://img.shields.io/pypi/dw/physai?color=blue&label=Weekly%20Downloads)
[![Socket Badge](https://badge.socket.dev/pypi/package/physai/5.1.0?artifact_id=tar-gz)](https://badge.socket.dev/pypi/package/physai/5.1.0?artifact_id=tar-gz)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17214724.svg)](https://doi.org/10.5281/zenodo.17214724)
[![▶ Open Demo Site](https://img.shields.io/badge/Site_&_Demo-View%20the%20site-FFDB3A)](https://ms-agi.github.io/PhysAI/)


> [Jump to Citation](#citation) | If you use PhysAI, neural operators, or its cross-validation tools in academic work, please cite [DOI 10.5281/zenodo.17214724](https://doi.org/10.5281/zenodo.17214724).

### Unified Operator Synthesis & Cross-Validation

PhysAI is an open-source, multi-backend framework for solving partial differential equations with Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNOs), and Unified Spectral Element architectures.

<table>
  <tr>
    <td><a href="https://github.com/MS-AGI/PhysAI#installation"><img src="https://img.shields.io/badge/Installation-Guide-4F46E5?style=for-the-badge" alt="Installation Guide"></a></td>
    <td><a href="https://github.com/MS-AGI/PhysAI#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Examples-0F766E?style=for-the-badge" alt="Quick Start examples"></a></td>
    <td><a href="https://ms-agi.github.io/PhysAI/"><img src="https://img.shields.io/badge/Live-Site-F59E0B?style=for-the-badge" alt="Live site"></a></td>
  </tr>
</table>


---

## Overview

**PhysAI** is a research library for approximating solutions to partial and ordinary differential equations with neural networks, built around the physics rather than around any single deep learning framework. It implements **Physics-Informed Neural Networks (PINNs)**, **Fourier Neural Operators (FNOs)**, and a **Unified Spectral Element Neural Operator (USENO)** on a common backend abstraction spanning **PyTorch, JAX, TensorFlow, and PaddlePaddle**, so the same governing equation, domain, and boundary/initial conditions train identically regardless of which deep learning framework a given lab, cluster, or paper already standardizes on.

The library is organized around the physics problem, not the network architecture: a **registry of 57 governing equations** — elliptic and parabolic PDEs, the compressible and incompressible Navier–Stokes and Euler systems, the linear and nonlinear Schrödinger equation, reaction–diffusion and pattern-formation systems, stochastic/kinetic (Fokker–Planck) equations, and relativistic and quantum-field residuals spanning the Dirac equation, the Einstein field equations, and quantum-gas statistics — an **SDF/CSG-based arbitrary-geometry system** for domains beyond a box or ball, a heuristic **AutoOptimizer** that reads the order, nonlinearity, and stiffness of a chosen equation to size the network and pick a training schedule, and **independent numerical cross-validation** through Dedalus, embedded-boundary finite differences, or optional native solver adapters.

### What's actually here

* **57 governing equations**, spanning elliptic/static problems, diffusion–reaction systems, hyperbolic/wave equations, nonlinear transport, fluid dynamics, quantum and dispersive systems, kinetic/probabilistic equations, excitable media, and relativistic/quantum-field theory (general relativity, the Dirac equation, quantum-gas statistics, phonon transport) — see [Governing Equations](#governing-equations) below, and [Planned Equations](#planned-equations) for what's coming next.
* **Three model architectures**: a Fourier-feature PINN (`physai.models.pinn`), a Fourier Neural Operator (`physai.models.fno`, following Li et al., 2020), and a Chebyshev-basis Unified Spectral Element Neural Operator (`physai.models.spectral_element` / `spectralpinn`, following the USENO formulation of Feugmo & Pankaczy) with C⁰ (value) and C¹ (flux) interface-continuity losses for stiff multiphysics problems.
* **`AutoOptimizer`**: reads equation order, nonlinearity, and stiffness to emit a frozen `RuntimeConfig` — learning rate, optimizer choice (Adam, with an optional L-BFGS fine-tuning phase), collocation-point density, residual/boundary loss weighting, warm-up/curriculum schedule, network width and depth, and float32 vs. float64 precision.
* **Arbitrary geometry via signed distance functions**: primitives (box, ball, cylinder, half-space, ellipsoid, torus, capsule, 2-D polygon), CSG combinators (`union`, `intersection`, `difference`, `smooth_union`, `invert`), user-defined SDFs as plain Python callables, and mesh import (`.stl`/`.obj`/`.ply`/`.off`) — with a `BoundaryConditionSet` attaching Dirichlet, Neumann, Robin, or periodic conditions to arbitrary regions of the boundary (`face_region`, `everywhere`, or a custom predicate).
* **Numerical cross-validation**, not only closed forms: `Trainer.cross_validate(...)` compares model outputs with independent Dedalus, embedded-boundary finite-difference, or optional native solver results. See [Numerical Cross-Validation](#numerical-cross-validation).
* **Physics-focused visualization and animation**: loss-history, residual-field, spectrum, and 1-D/2-D solution plots, plus a dedicated N-dimensional toolkit — slicing, projection, volumetric isosurface rendering, and time or parameter-sweep animation — for fields with three, four, or more axes. See [Visualization and Animation](#visualization-and-animation).
* **Multi-backend by construction**, not by wrapping one framework: `AbstractBackend` fixes the tensor/autodiff/optimizer surface, and `TorchBackend`, `JAXBackend`, `TensorFlowBackend`, and `PaddleBackend` each implement it, so residuals and losses are written once against the abstraction and run correctly on all four.
* **Optional, consent-gated extras**: a live terminal training dashboard (`rich`) with an optional local-LLM chat side panel (`llama-cpp-python`); completed runs also produce a PDF with total and per-term loss curves and a target-convergence summary (`physai_training_report.pdf` by default). A bundled Conda installer provides Dedalus, FiPy, FEniCS, FEniCSx, Meep, and CuPy — nothing installs on `pip install`/`import physai`; setup is user-invoked and stays inert in CI/headless environments.

---

## Installation

```bash
pip install physai
```

Backend and feature extras are opt-in, so a base install stays light (torch + numpy + matplotlib only):

```bash
pip install "physai[jax]"          # JAX backend (jax, flax, optax)
pip install "physai[tensorflow]"   # TensorFlow backend
pip install "physai[paddle]"       # PaddlePaddle backend
pip install "physai[dashboard]"    # live terminal training dashboard (rich)
pip install "physai[chat]"         # dashboard + local GGUF chat side panel (llama-cpp-python)
pip install "physai[all]"          # every backend and feature extra above
```

Or, from source:

```bash
git clone https://github.com/MS-AGI/PhysAI.git
cd PhysAI
pip install -e ".[jax,tensorflow,dashboard]"
```

Python ≥ 3.9. Optional native solvers use compiled dependencies such as MPI and PETSc, so the solver stack is installed with Conda rather than as a `pyproject.toml` extra. `physai.install_solver_dependencies()` or `python -m physai.solver_setup` runs the bundled installer, which creates a separate `physai-solvers` Conda environment. Activate that environment, install PhysAI there with `python -m pip install physai`, and run scripts from it to use those solver adapters.

> **JAX users:** install via the `jax` extra (or `requirements.txt`) rather than an unpinned `pip install jax flax` — see [Backend Notes](#backend-notes) for why the pin matters.

---

## Repository Structure

```
src/physai/
├── core/
│   ├── pde_residual.py      # PDE_REGISTRY: 57 governing-equation residuals
│   ├── auto_optimizer.py    # ProblemSpec -> AutoOptimizer -> RuntimeConfig
│   └── losses.py            # residual / dirichlet / neumann / robin / periodic losses
├── backends/                 # AbstractBackend + torch / jax / tensorflow / paddle
├── models/
│   ├── pinn.py               # Fourier-feature PINN, hard-constraint support
│   ├── fno.py                 # Fourier Neural Operator
│   └── spectral_element.py, spectralpinn.py   # USENO (Chebyshev spectral element)
├── solvers/
│   └── solver.py             # Dedalus, embedded-boundary FD, and optional native solver adapters
├── geometry.py                # SDF primitives, CSG, mesh import, BoundaryConditionSet
├── trainer.py                 # Trainer: training loop, callbacks, cross_validate, per-backend step logic
├── visualization.py            # 1-D/2-D plots, loss curves, spectra, animations
├── visualization_nd.py          # slicing / projection / isosurfaces / animation for N-D fields
├── dashboard/live.py            # optional live terminal dashboard (Callback)
├── chat_setup.py, solver_setup.py   # consent-gated optional solver setup
└── utils.py                    # sampling (LHS/Sobol), metrics, seeding, dtype helpers
tests/
└── test_pde_everything.py     # Tier A/B/C suite — see Testing below
examples/
README.md
pyproject.toml
requirements.txt
```

---

## Quick Start

These runnable examples show a user-registered Maxwell residual and animation, a wave PINN cross-validated against Dedalus, and a PINN trained with the Einstein vacuum residual using Schwarzschild exterior metric data. Run commands from the repository root after installing PhysAI and the relevant dependencies.

### 1. Register Maxwell's equations and animate the field

This example registers a one-dimensional vacuum Maxwell system with `register_pde`, trains six field outputs, and animates the transverse electric field `Ey`. See [examples/maxwell_animation.py](examples/maxwell_animation.py).

~~~bash
python examples/maxwell_animation.py
~~~

### 2. Solve the wave equation and cross-validate

This example trains the first-order state `(u, v)` for the wave equation and compares both fields with an independent Dedalus solve. It requires the Conda solver environment: run `python -m physai.solver_setup`, activate `physai-solvers`, install PhysAI there with `python -m pip install physai`, then run the command below. See [examples/wave_cross_validation.py](examples/wave_cross_validation.py).

~~~bash
python examples/wave_cross_validation.py
~~~

### 3. Fit a Schwarzschild exterior with the Einstein field residual

This example trains `einstein_field` against the exact isotropic-coordinate Schwarzschild vacuum metric on a spatial region outside the horizon. It is a Schwarzschild exterior spacetime example, not a cosmological model. See [examples/schwarzschild_einstein_residual.py](examples/schwarzschild_einstein_residual.py).

~~~bash
python examples/schwarzschild_einstein_residual.py
~~~

One JAX-specific step for other scripts: because JAX/Flax keep parameters outside the model object rather than on it, call trainer.init_jax(dummy_input) once after constructing Trainer and before trainer.train() when using the JAX backend.

---
## Governing Equations

`physai.core.pde_residual.PDE_REGISTRY` currently implements 57 residuals (`_PDE_META` in `auto_optimizer.py` records each equation's order, nonlinearity, and stiffness for `AutoOptimizer`'s heuristics):

| Category | Equations |
| --- | --- |
| Elliptic / static | `poisson`, `laplace`, `helmholtz`, `biharmonic`, `biharmonic_steady`, `darcy`, `brinkman_darcy` |
| Parabolic / diffusion–reaction | `heat`, `diffusion`, `reaction_diffusion`, `fisher_kpp`, `allen_cahn`, `cahn_hilliard`, `cahn_hilliard_2d`, `porous_medium`, `perona_malik`, `phase_field_crystal`, `swift_hohenberg`, `gray_scott`, `gierer_meinhardt` |
| Hyperbolic / wave | `wave`, `viscous_wave`, `klein_gordon`, `klein_gordon_nonlinear_2d`, `sine_gordon_2d`, `boussinesq_wave`, `euler_tricomi`, `regularised_long_wave` |
| Nonlinear transport | `advection`, `burgers`, `burgers_2d`, `kdv`, `kuramoto_sivashinsky`, `kadomtsev_petviashvili` |
| Fluid dynamics | `navier_stokes`, `euler`, `stokes`, `shallow_water_2d`, `boussinesq_convection`, `relativistic_fluid` |
| Quantum / dispersive | `schrodinger`, `nls`, `nlse_2d`, `complex_ginzburg_landau_2d`, `radhakrishnan_kundu_lakshmanan` |
| Kinetic / probabilistic | `fokker_planck`, `fokker_planck_2d`, `drift_diffusion_poisson` |
| Excitable media / pattern formation | `fitzhugh_nagumo`, `dendritic_solidification` |
| First-order / geometric | `eikonal` |
| **Relativistic & quantum-field theory** | `einstein_field` (Einstein field equations, symmetry-reduced), `dirac` (relativistic spin-½ wave equation), `bose_einstein` (Gross–Pitaevskii / BEC), `fermi_gas` (Fermi–Dirac quantum-gas statistics), `phonon` (lattice-vibration dispersion and thermal transport), `quantum_relativistic_fluid` (relativistic hydrodynamics with a quantum-corrected equation of state) |
| Auxiliary: mathematical finance | `black_scholes_2d`, `heston_volatility` — included because the Black–Scholes and Heston PDEs share the same parabolic/elliptic residual machinery as the rest of the registry, not a primary focus of the library |

Composite/coupled multi-physics residuals built from the equations above are also available — see the "Mixed / composite residuals" section of `pde_residual.py`.

### Planned Equations

Equations under active development for a future release, extending the library's coverage of classical and quantum field theory:

* **Solitons** — a general soliton-solution residual framework, beyond the KdV- and NLS-family solitons already covered by the existing `kdv` and `nls` equations.
* **Maxwell's Equations** — the full coupled electromagnetic field system, extending the library's current electromagnetism coverage beyond the drift–diffusion/Poisson treatment in `drift_diffusion_poisson`.
* **Group Field Theory** — residuals for group field theory models, relevant to quantum-gravity and quantum-gravity-adjacent research.

The planned Maxwell item refers to a built-in full-system residual; the examples include a user-registered one-dimensional vacuum reduction. This list reflects current development priorities and is not a commitment to a specific release date. Contributions and equation requests are welcome via GitHub Discussions and Issues.

---

## Numerical Cross-Validation

Beyond checking a trained network against a closed-form solution (available for only a curated subset of equations), `Trainer.cross_validate(...)` compares it with an **independent classical numerical solve** through `physai.solvers.solver.Solver`. The model is evaluated on the classical solver's coordinates, and the method reports absolute and relative L2 errors. It supports three paths:

* **Box domains** (`geometry=None`, the default) — a Dedalus spectral solve (`Solver.solve_box`) using tensor-product Chebyshev/Fourier bases. It requires `domain_type`, `bounds`, `variables`, `equations`, `bcs`, and `ics`; the axis settings accept one value for all axes or a per-axis list. Install the Conda solver stack, activate `physai-solvers`, and install PhysAI in that environment before running this path.
* **Arbitrary geometry and PDE residuals** (`geometry=<a physai.geometry.Geometry>`) — `Solver.solve_geometry` keeps sparse embedded-boundary finite-difference fast paths for Poisson, Helmholtz, and heat, and routes other registered PDEs or caller-supplied residuals through `autosolve`. AutoSolve works from the `residual(model_fn, points)` contract on boxes, CSG, smooth SDFs, and mesh geometries; unregistered equations can supply a residual callable with their constraints.
* **General classical discretizations**: `Solver.solve(method="discrete_geometry", geometry=..., ...)` solves a sparse linear system or nonlinear residual/Jacobian system on any `Geometry`, including CSG and smooth SDF shapes. Supply `assembler(context) -> (A, b)` or `residual_fn(u, context)` plus `jacobian_fn(u, context)`. The context includes the masked grid, compact unknown indices, projected boundary crossings, normals, and optional space-time coordinates. This handles any PDE for which you provide a classical discretization; a PDE name alone does not determine a numerical scheme.
* **Unregistered PDE residuals**: `autosolve(residual, geometry, ...)` accepts any callable with the existing `residual(model_fn, points) -> residual tensor` contract, including residual classes not in `PDE_REGISTRY`. It uses a smooth Gaussian RBF field and SciPy nonlinear least squares, with geometry-aware interior/boundary samples, Dirichlet/Neumann/Robin/periodic constraints, optional initial/data constraints, and an `AutoSolverOptimizer` that budgets collocation points from geometry, output count, and available PDE metadata. The optimizer scales anisotropic space-time coordinates, balances PDE and constraint residual blocks, and selects extra RBF regularization when the kernel is ill-conditioned. For a new coupled system, pass `n_output`; the residual must support the selected backend's differentiation operations. RBF collocation is a general method, while convergence still depends on the PDE, constraints, smoothness, and resolution. Example: `solution = autosolve(my_residual, geometry, backend="torch", n_output=2, boundary_conditions=bcs)`.
* **LaTeX equations into AutoSolve**: `build_latex_residual(...)` validates supported syntax and returns a backend-differentiable residual that can be passed straight to `autosolve(residual, geometry, ...)`. Or call `register_latex_pde(...)` and let `autosolve(geometry=..., pde="name", ...)` build the registered residual. The parser supports arithmetic, common scalar functions, partial derivative fractions/subscripts, and scalar Laplacians; declare field/coordinate order explicitly. AutoSolve still needs a geometry and suitable boundary, initial, or data constraints for the problem.
* **Native solver adapters** — pass `solver_method="fipy"`, `"fenics"`, `"fenicsx"`, or `"meep"` and its native arguments through `solver_kwargs`. For a custom adapter, register it with `register_solver`; `register_equation_solver` can associate a PDE name with a solver. FiPy and scalar finite-element results are normalized automatically. For Meep or a custom result format, pass `classical_result_adapter` that returns coordinates and named values. These packages are installed in the same Conda environment by `physai.install_solver_dependencies()`.

```python
metrics = trainer.cross_validate(
    domain_type="chebyshev",
    bounds=(0.0, 1.0),
    variables=["u"],
    equations=["dt(u) - dx(dx(u)) = 0"],
    bcs=["left(u) = 0", "right(u) = 0"],
    ics={"u": lambda x: np.sin(np.pi * x)},
    stop_time=0.1,
    grid_points=128,
)
# {"u_l2_abs": ..., "u_l2_rel": ..., "u_n_compared": ...}
```

Nothing about training is touched by calling `cross_validate` — it runs the classical solve independently, evaluates the trained model at the same grid points, and returns the comparison.

### Register a PDE from LaTeX and solve it with AutoSolve

LaTeX registration turns a supported equation into the same callable residual interface used by registered and custom PDEs. AutoSolve can build that residual by registry name, or accept the compiled residual directly. You still provide the domain and the constraints that make the problem well-posed; syntax validation does not infer boundary or initial conditions.

```python
import numpy as np

from physai import BoundaryConditionSet, autosolve, box, register_latex_pde
from physai.backends import get_backend

backend = get_backend("torch")
equation = (
    r"\frac{\partial u}{\partial t} "
    r"+ u\frac{\partial u}{\partial x} "
    r"- \nu\frac{\partial^2 u}{\partial x^2} = 0"
)
register_latex_pde(
    "custom_burgers",
    equation,
    fields=("u",),
    coordinates=("x", "t"),
    parameters={"nu": 0.01},
)

geometry = box([(-1.0, 1.0)])
boundary_conditions = BoundaryConditionSet(geometry)
for endpoint in (-1.0, 1.0):
    boundary_conditions.add(
        "dirichlet",
        value=lambda points: np.zeros(len(points)),
        region=lambda points, endpoint=endpoint: np.isclose(points[:, 0], endpoint),
        name=f"x_{endpoint}",
    )

x0 = np.linspace(-1.0, 1.0, 32)
ic_points = np.column_stack((x0, np.zeros_like(x0)))
solution = autosolve(
    geometry=geometry,
    pde="custom_burgers",
    backend=backend,
    time_domain=(0.0, 1.0),
    boundary_conditions=boundary_conditions,
    ic_points=ic_points,
    ic_values=-np.sin(np.pi * x0),
)
```

For a one-off equation, call `build_latex_residual(equation, backend, fields=..., coordinates=..., parameters=...)` and pass the returned object as the first argument to `autosolve(residual, geometry, ...)`. The current LaTeX parser accepts its documented scalar expression grammar; it does not interpret arbitrary LaTeX commands or tensor notation. AutoSolve convergence depends on the equation, constraints, geometry sampling, and resolution.

---

## Testing
**The testing logs with coverage is present in `tests/tests_log.txt`**.

The end-to-end suite lives in `tests/` and runs across every backend whose underlying framework is importable in the current environment (a missing framework is skipped, not a collection failure):

* **Tier A** — every one of the 57 registered equations trains for a few steps on a simple domain and is checked for finite, non-diverging loss. A mechanical pipeline test (geometry sampling, BC/IC wiring, `AutoOptimizer` sizing, and the training loop all run), not a convergence claim.
* **Tier B** — a curated subset with independently hand-verified closed-form solutions, trained on a deliberately harder off-center domain with mixed Dirichlet/Neumann boundary conditions, and checked against the analytic solution on held-out interior points. A mechanical counterpart of the same hard geometry/BC/IC also runs across the full 57-equation registry, without requiring a known solution.
* **Tier C** — cross-validation against real Dedalus spectral-solver output for a small curated subset; skipped automatically when Dedalus is not installed in the active environment.

```bash
pytest tests/test_pde_everything.py -v
```

---

## Backend Notes

PhysAI abstracts tensors, autodiff, and optimizers behind `physai.backends.base.AbstractBackend`, implemented by `TorchBackend`, `JAXBackend`, `TensorFlowBackend`, and `PaddleBackend`. A few backend-specific points:

* **JAX**: parameters live outside the model object (Flax-style), so `Trainer.init_jax(dummy_input)` must be called once before `trainer.train()` — see the Quick Start example.
* **JAX/Flax version pin**: install via `pip install "physai[jax]"` or `requirements.txt` rather than an unpinned `jax`/`flax`. JAX ≥ 0.11 removes an internal API (`jax.core.get_opaque_trace_state`) that older Flax releases still call, which surfaces as an `AttributeError` inside `trainer.init_jax(...)`. The `jax<0.11`/`flax>=0.10,<0.11` pins in `pyproject.toml` keep the pair compatible.
* **Precision**: `AutoOptimizer` recommends float64 for equations flagged stiff in `_PDE_META`, and float32 otherwise; override via `ProblemSpec.extra_params` if a given problem needs a different precision than the heuristic selects.

---

## Visualization and Animation

`physai.visualization` covers 1-D and 2-D fields — the shape most PDE solutions are inspected in during development:

```python
from physai.visualization import (
    plot_loss_history, plot_solution_1d, plot_solution_2d,
    plot_solution_2d_comparison, plot_residual_field,
    plot_collocation_points, plot_spectrum,
    animate_1d_solution, plot_training_animation, plot_error_convergence,
)
```

* `plot_loss_history` — residual/BC/IC loss terms on a shared log scale.
* `plot_solution_2d_comparison` — prediction, reference, and pointwise error side by side.
* `plot_residual_field` — the spatial distribution of the PDE residual itself, useful for locating where a trained network is furthest from satisfying the governing equation.
* `plot_spectrum` — the FFT-based energy spectrum of a predicted field, for checking whether a network has captured the expected frequency content (relevant for, e.g., turbulent or dispersive solutions).

`physai.visualization_nd` handles fields of three or more axes — 3-D volumes, 4-D spatio-temporal fields, or higher-dimensional parameter sweeps — via slicing, projection, and animation, working from a plain `numpy.ndarray` plus coordinate arrays for each axis, independent of the originating PDE's dimensionality:

```python
from physai.visualization_nd import (
    describe_field, slice_field, project_field,
    plot_slice, plot_slice_grid, plot_isosurface_3d,
    animate_nd_field, animate_isosurface_3d, interactive_nd_explorer,
)
```

* `slice_field` / `plot_slice` — fix all but one or two axes at a given index and inspect the resulting 1-D/2-D cross-section.
* `project_field` — reduce extra axes with mean, max, sum, or RMS aggregation, collapsing an N-D field to something directly plottable.
* `plot_isosurface_3d` / `animate_isosurface_3d` — volumetric isosurface rendering for 3-D scalar fields, via `plotly` when installed, with a matplotlib voxel/scatter fallback otherwise.
* `animate_nd_field` — sweep one axis (time, a physical parameter, or a spatial slice index) as an animation, holding or projecting the remaining axes.
* `interactive_nd_explorer` — an interactive slicing/projection widget for exploratory inspection of a solution field.

All plotting functions return `(fig, axes)` and never call `plt.show()`, so display and saving remain under the caller's control.

---

## Citation
If you use **PhysAI** in your research, academic publication, or official work, **citation is required**.

Please cite the software as follows:

**APA:**
> Singh, M. ([https://orcid.org/0009-0009-3913-6929](https://orcid.org/0009-0009-3913-6929)) (2026). *PhysAI: A Multi-Backend Physics-Informed Neural Network Framework for Solving, Cross-Validating, and Visualizing Ordinary and Partial Differential Equations* (Version 5.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17214724

**BibTeX:**
```bibtex
@software{singh_physai_2026,
  author       = {Mankrit Singh},
  title        = {PhysAI: A Multi-Backend Physics-Informed Neural Network Framework for Solving, Cross-Validating, and Visualizing Ordinary and Partial Differential Equations},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {5.1.0},
  doi          = {10.5281/zenodo.17214724},
  url          = {https://doi.org/10.5281/zenodo.17214724},
  orcid        = {0009-0009-3913-6929}
}
```

---

## License

AGPL-3.0 License. See [LICENSE](LICENSE) file.
