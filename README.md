# PhysAI: A Multi-Backend Physics-Informed Neural Network Library for PDE Solving at Research Scale


[![PyPI version](https://img.shields.io/pypi/v/physai.svg)](https://pypi.org/project/physai/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/AGPL_License_3.0-indigo.svg)](https://opensource.org/licenses/agpl-3-0)
![PyPI - Total Downloads](https://img.shields.io/pypi/dw/physai?color=blue&label=Weekly%20Downloads)
[![Socket Badge](https://badge.socket.dev/pypi/package/physai/4.0.0?artifact_id=tar-gz)](https://badge.socket.dev/pypi/package/physai/4.0.0?artifact_id=tar-gz)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17493878.svg)](https://doi.org/10.5281/zenodo.17493878)
[![▶ Open Demo Site](https://img.shields.io/badge/Site_&_Demo-View%20the%20site-FFDB3A)](https://ms-agi.github.io/PhysAI/)

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PhysAI: A Multi-Backend Physics-Informed Neural Network Library for PDE Solving at Research Scale</title>
  <style>
    /* Core Section Setup */
    .attention-hero {
      position: relative;
      background-color: #030712;
      color: #f3f4f6;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 8rem 2rem;
      display: flex;
      justify-content: center;
      align-items: center;
      text-align: center;
      overflow: hidden;
    }

    /* Ambient Glow Effects */
    .hero-glow-1, .hero-glow-2 {
      position: absolute;
      width: 400px;
      height: 400px;
      border-radius: 50%;
      filter: blur(120px);
      opacity: 0.15;
      z-index: 1;
      pointer-events: none;
    }
    .hero-glow-1 { background: #3b82f6; top: -10%; left: 20%; }
    .hero-glow-2 { background: #8b5cf6; bottom: -10%; right: 20%; }

    .hero-content {
      position: relative;
      z-index: 2;
      max-width: 850px;
      margin: 0 auto;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      padding: 0.5rem 1rem;
      background: rgba(98, 96, 96, 0.31);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 9999px;
      font-weight: 900;
      color: #9a78f0;
      margin-bottom: 2rem;
      transition: all 0.2s ease;
    }
    .badge:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .hero-content h1 {
      font-size: 3.5rem;
      font-weight: 800;
      line-height: 1.2;
      letter-spacing: -0.02em;
      margin-bottom: 1.5rem;
    }
    @media (max-width: 768px) {
      .hero-content h1 { font-size: 2.5rem; }
    }

    .text-gradient {
      background: linear-gradient(135deg, #9a91fb, #a13ef8, #5706d1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero-content p {
      font-size: 1.15rem;
      color: #9ca3af;
      line-height: 1.6;
      max-width: 720px;
      margin: 0 auto 2.5rem auto;
    }

    /* Citation Callout Note */
    .citation-box {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      height: 125;
      width: 100%;
      padding: 0.5rem 0.5rem;
      
      background: rgba(137, 4, 4, 0.27);
      border: 2px solid rgba(255, 176, 176, 0.95);
      border-radius: 25px;
      font-weight: 500;
      color: #050114;
      margin-bottom: 2rem;
      animation: activeGlow 2s infinite ease-in-out;
    }

  /* 2. Define the glowing animation keyframes */
    @keyframes activeGlow {
      0% {
        box-shadow: 0 0 5px rgba(245, 158, 11, 0.2);
        border-color: #f59e0b;
      }
      50% {
        /* The peak of the glow: broader spread and brighter border */
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.6);
        border-color: #fbbf24;
      }
      100% {
        box-shadow: 0 0 5px rgba(245, 158, 11, 0.2);
        border-color: #f59e0b;
      }
    }
    .citation-box:hover {
      background: rgba(243, 108, 108, 0.18)
    }
    .citation-note {
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: 0.85rem;
      color: #6b7280;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      padding-top: 1.5rem;
    }
    .citation-note code {
      background: rgba(255, 255, 255, 0.05);
      padding: 0.125rem 0.25rem;
      border-radius: 0.25rem;
      color: #d1d5db;
    }

    /* ─── FIXED BUTTONS ENGINE ─── */
    .cta-group {
      display: flex;
      gap: 1rem;
      justify-content: center;
      align-items: center;
    }
    
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.625rem;
      white-space: nowrap;
      padding: 0.875rem 2rem;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 0.75rem;
      text-decoration: none;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      box-sizing: border-box;
    }

    .btn-primary {
      background-color: #ffffff;
      color: #0f172a;
      box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05), 0px 0px 20px rgba(99, 102, 241, 0.25);
    }
    
    .btn-primary:hover {
      background-color: #f3f4f6;
      transform: translateY(-2px);
      box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15), 0px 0px 30px rgba(99, 102, 241, 0.45);
    }
    
    .btn-arrow {
      display: inline-block;
      flex-shrink: 0;
      transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .btn-primary:hover .btn-arrow {
      transform: translateX(4px);
    }

    .btn-secondary {
      background-color: transparent;
      color: #ffffff;
      border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .btn-secondary:hover {
      background-color: rgba(255, 255, 255, 0.05);
      border-color: rgba(255, 255, 255, 0.35);
    }

    /* Responsive Stack Breakpoint */
    @media (max-width: 540px) {
      .cta-group { 
        flex-direction: column; 
        width: 100%;
        padding: 0 1rem;
      }
      .btn { 
        width: 100%; 
      }
    }
  </style>
</head>
<body>

<section class="attention-hero">
  <div class="hero-glow-1"></div>
  <div class="hero-glow-2"></div>
  
  <div class="hero-content">
    <!-- Requested Citation Notice Component -->
    <a href="#citation" class="citation-box">
    <div class="citation-note">
      <p>If you leverage this library, neural operators, or cross-validation pipelines in your academic publications, please cite our repository or reference the official Zenodo DOI record: <code>10.5281/zenodo.17493878</code>.</p>
    </div>
    </a>
    <!-- Inline Flex Centered Badge Component -->
    <a href="#testing" class="badge" style="height: 50px; width: 250px; font-size: 20px; display: inline-flex; align-items: center; justify-content: center;">Rigorously Tested</a>
    <!-- Academic Headline String Integration -->
    <h1>
      PhysAI: <span class="text-gradient">Unified Operator Synthesis & Cross-Validation</span>
    </h1>
    <!-- Formal & Academic Description Replacing Former Boilerplate Placeholders -->
    <p>
      An open-source, multi-backend computing framework implementing Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNOs), and Unified Spectral Element architectures. Streamline continuous partial differential equation solvers with invariant performance layers across PyTorch, JAX,Paddle  and TensorFlow targets.
    </p>
    <div class="cta-group">
      <a href="#installation" class="btn btn-primary">
        Installation Guide
        <svg class="btn-arrow" xmlns="http://w3.org" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      </a>
      <a href="#quick-start" class="btn btn-secondary">Quick Start</a>
    </div>

  </div>
</section>

</body>
</html>

---

## Overview

**PhysAI** is a research library for approximating solutions to partial and ordinary differential equations with neural networks, built around the physics rather than around any single deep learning framework. It implements **Physics-Informed Neural Networks (PINNs)**, **Fourier Neural Operators (FNOs)**, and a **Unified Spectral Element Neural Operator (USENO)** on a common backend abstraction spanning **PyTorch, JAX, TensorFlow, and PaddlePaddle**, so the same governing equation, domain, and boundary/initial conditions train identically regardless of which deep learning framework a given lab, cluster, or paper already standardizes on.

The library is organized around the physics problem, not the network architecture: a **registry of 57 governing equations** — elliptic and parabolic PDEs, the compressible and incompressible Navier–Stokes and Euler systems, the linear and nonlinear Schrödinger equation, reaction–diffusion and pattern-formation systems, stochastic/kinetic (Fokker–Planck) equations, and relativistic and quantum-field residuals spanning the Dirac equation, the Einstein field equations, and quantum-gas statistics — an **SDF/CSG-based arbitrary-geometry system** for domains beyond a box or ball, a heuristic **AutoOptimizer** that reads the order, nonlinearity, and stiffness of a chosen equation to size the network and pick a training schedule, and a **numerical cross-validation path against Dedalus**, so a trained network's error can be checked against a genuine independent solve of the same equation, not only against a closed-form solution when one happens to exist.

### What's actually here

* **57 governing equations**, spanning elliptic/static problems, diffusion–reaction systems, hyperbolic/wave equations, nonlinear transport, fluid dynamics, quantum and dispersive systems, kinetic/probabilistic equations, excitable media, and relativistic/quantum-field theory (general relativity, the Dirac equation, quantum-gas statistics, phonon transport) — see [Governing Equations](#governing-equations) below, and [Planned Equations](#planned-equations) for what's coming next.
* **Three model architectures**: a Fourier-feature PINN (`physai.models.pinn`), a Fourier Neural Operator (`physai.models.fno`, following Li et al., 2020), and a Chebyshev-basis Unified Spectral Element Neural Operator (`physai.models.spectral_element` / `spectralpinn`, following the USENO formulation of Feugmo & Pankaczy) with C⁰ (value) and C¹ (flux) interface-continuity losses for stiff multiphysics problems.
* **`AutoOptimizer`**: reads equation order, nonlinearity, and stiffness to emit a frozen `RuntimeConfig` — learning rate, optimizer choice (Adam, with an optional L-BFGS fine-tuning phase), collocation-point density, residual/boundary loss weighting, warm-up/curriculum schedule, network width and depth, and float32 vs. float64 precision.
* **Arbitrary geometry via signed distance functions**: primitives (box, ball, cylinder, half-space, ellipsoid, torus, capsule, 2-D polygon), CSG combinators (`union`, `intersection`, `difference`, `smooth_union`, `invert`), user-defined SDFs as plain Python callables, and mesh import (`.stl`/`.obj`/`.ply`/`.off`) — with a `BoundaryConditionSet` attaching Dirichlet, Neumann, Robin, or periodic conditions to arbitrary regions of the boundary (`face_region`, `everywhere`, or a custom predicate).
* **Numerical cross-validation against Dedalus**, not only closed forms: `Trainer.cross_validate(...)` runs an independent classical solve of the same equation — a genuine Dedalus spectral solve (tensor-product Chebyshev/Fourier bases, box domains) or an embedded-boundary finite-difference solve on the same arbitrary geometry the network was trained on — and reports the L2 error between the two, on the same evaluation grid. See [Numerical Cross-Validation](#numerical-cross-validation).
* **Physics-focused visualization and animation**: loss-history, residual-field, spectrum, and 1-D/2-D solution plots, plus a dedicated N-dimensional toolkit — slicing, projection, volumetric isosurface rendering, and time or parameter-sweep animation — for fields with three, four, or more axes. See [Visualization and Animation](#visualization-and-animation).
* **Multi-backend by construction**, not by wrapping one framework: `AbstractBackend` fixes the tensor/autodiff/optimizer surface, and `TorchBackend`, `JAXBackend`, `TensorFlowBackend`, and `PaddleBackend` each implement it, so residuals and losses are written once against the abstraction and run correctly on all four.
* **Optional, consent-gated extras**: a live terminal training dashboard (`rich`) with an optional local-LLM chat side panel (`llama-cpp-python`), and a bundled Dedalus installer — none of these run, prompt, or download anything on `pip install`/`import physai`; they act only on explicit, per-run consent, and stay entirely inert in CI/headless environments.

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

Python ≥ 3.9. Dedalus, used for numerical cross-validation, is **not** pip-installable — it requires a Conda environment with MPI/FFTW — and is therefore not a `pyproject.toml` extra at all. `physai.install_dedalus()` runs a bundled cross-platform installer on explicit request; PhysAI otherwise imports and trains fully without it.

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
│   └── dedalus.py            # real Dedalus (box) + embedded-boundary FD (arbitrary geometry)
├── geometry.py                # SDF primitives, CSG, mesh import, BoundaryConditionSet
├── trainer.py                 # Trainer: training loop, callbacks, cross_validate, per-backend step logic
├── visualization.py            # 1-D/2-D plots, loss curves, spectra, animations
├── visualization_nd.py          # slicing / projection / isosurfaces / animation for N-D fields
├── dashboard/live.py            # optional live terminal dashboard (Callback)
├── chat_setup.py, dedalus_setup.py  # consent-gated optional-extra setup
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

PhysAI's entry point is `AutoOptimizer`, not a hand-built model and training loop: the governing equation and domain are declared via a `ProblemSpec`, and `AutoOptimizer` emits a `RuntimeConfig` that `Trainer` consumes directly.

```python
import numpy as np
from physai.backends import get_backend
from physai.core.auto_optimizer import ProblemSpec, AutoOptimizer
from physai.core.pde_residual import build_residual
from physai.geometry import DomainSpec, box, BoundaryConditionSet, everywhere
from physai.trainer import Trainer

backend = get_backend("torch", device="cpu") # Simple API style backend- change with a single string change!

# 2-D Poisson equation, ∇²u = f, on the unit box, with a manufactured
# solution u(x, y) = x² + y² imposed as Dirichlet data on the boundary.
domain = DomainSpec(spatial_dims=2, bounds=[(-1, 1), (-1, 1)])
analytic = lambda x: (x[:, 0] ** 2 + x[:, 1] ** 2).reshape(-1, 1)

geom = box(domain.bounds)
bcs = BoundaryConditionSet(geom)
bcs.add("dirichlet", value=lambda x: analytic(x).astype(np.float32), region=everywhere)

spec = ProblemSpec(pde_name="poisson", domain=domain, backend_name="torch")
config = AutoOptimizer(backend, verbose=True).analyse(spec)          # sizes the network, picks lr/optimizer/etc.
residual = build_residual("poisson", backend)

rng = np.random.default_rng(0)
coll = backend.tensor(rng.uniform(-1, 1, size=(2048, 2)).astype(np.float32))

trainer = Trainer(
    backend=backend, config=config, residual=residual,
    collocation_points=coll, boundary_conditions=bcs,
)
history = trainer.train()
```

Swap `TorchBackend()` for `JAXBackend()`, `TensorFlowBackend()`, or `PaddleBackend()` and the rest of the script is unchanged — `AutoOptimizer` reads `spec.backend_name` for backend-specific choices such as default precision. **One JAX-specific step:** because JAX/Flax keep parameters outside the model object rather than on it, call `trainer.init_jax(dummy_input)` once, immediately after constructing `Trainer` and before `trainer.train()`, whenever `backend.name == "jax"`.

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

This list reflects current development priorities and is not a commitment to a specific release date. Contributions and equation requests are welcome via GitHub Discussions and Issues.

---

## Numerical Cross-Validation

Beyond checking a trained network against a closed-form solution (available for only a curated subset of equations), `Trainer.cross_validate(...)` checks it against an **independent classical numerical solve** of the same governing equation, run by `physai.solvers.dedalus.Solver`, and reports the L2 error between the two on a shared evaluation grid. Two solve paths are dispatched from the same method:

* **Box domains** (`geometry=None`, the default) — a genuine [Dedalus](https://dedalus-project.org/) spectral solve (`Solver.solve_box`): a tensor-product Chebyshev/Fourier basis per axis, Dedalus's own IVP solver and tau-correction compiler, and its own independent timestepper and discretization. Requires `domain_type`, `bounds`, `variables`, `equations`, `bcs`, and `ics`; `domain_type`/`bounds`/`grid_points` each accept either a single value (applied to every axis) or a per-axis list, so this path is not limited to one spatial dimension.
* **Arbitrary geometry** (`geometry=<a physai.geometry.Geometry>`) — an embedded-boundary finite-difference solve (`Solver.solve_geometry`) on a masked regular N-D grid, assembled with `scipy.sparse` (or `cupyx.scipy.sparse` on GPU, if `array_module="cupy"`). Dedalus's own spectral bases are built for boxes and specific curvilinear coordinate systems, not arbitrary CSG/SDF domains, so this path exists specifically for geometries Dedalus can't represent. **Currently scoped to `"poisson"`, `"helmholtz"`, and `"heat"`**.

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
# {"u_l2_abs": ..., "u_l2_rel": ...}
```

Nothing about training is touched by calling `cross_validate` — it runs the classical solve independently, evaluates the trained model at the same grid points, and returns the comparison.

---

## Testing
**The testing logs with coverage is present in `tests/tests_log.txt`**.

The end-to-end suite lives in `tests/` and runs across every backend whose underlying framework is importable in the current environment (a missing framework is skipped, not a collection failure):

* **Tier A** — every one of the 57 registered equations trains for a few steps on a simple domain and is checked for finite, non-diverging loss. A mechanical pipeline test (geometry sampling, BC/IC wiring, `AutoOptimizer` sizing, and the training loop all run), not a convergence claim.
* **Tier B** — a curated subset with independently hand-verified closed-form solutions, trained on a deliberately harder off-center domain with mixed Dirichlet/Neumann boundary conditions, and checked against the analytic solution on held-out interior points. A mechanical counterpart of the same hard geometry/BC/IC also runs across the full 57-equation registry, without requiring a known solution.
* **Tier C** — cross-validation against real Dedalus spectral-solver output for a small curated subset; skipped automatically if `dedalus` isn't installed.

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
> Singh, M. ([https://orcid.org/0009-0009-3913-6929](https://orcid.org/0009-0009-3913-6929)) (2026). *PhysAI: A Multi-Backend Physics-Informed Neural Network Framework for Solving, Cross-Validating, and Visualizing Ordinary and Partial Differential Equations* (Version 5.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17493878

**BibTeX:**
```bibtex
@software{singh_physai_2026,
  author       = {Mankrit Singh},
  title        = {PhysAI: A Multi-Backend Physics-Informed Neural Network Framework for Solving, Cross-Validating, and Visualizing Ordinary and Partial Differential Equations},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {5.0.0},
  doi          = {10.5281/zenodo.17493878},
  url          = {https://doi.org/10.5281/zenodo.17493878},
  orcid        = {0009-0009-3913-6929}
}
```

---

## License

AGPL-3.0 License. See [LICENSE](LICENSE) file.