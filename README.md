# PhysAI: Modular PyTorch Library with Built-In Residual PDEs, Automatic PINN Losses, and Visualization Tools for Complex ODE/PDE Solving

[![PyPI version](https://img.shields.io/pypi/v/physai.svg)](https://pypi.org/project/physai/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/physai.svg)](https://pypi.org/project/physai/)
[![Socket Badge](https://badge.socket.dev/pypi/package/physai/2.5.7?artifact_id=tar-gz)](https://badge.socket.dev/pypi/package/physai/2.5.7?artifact_id=tar-gz)
[![DOI](https://zenodo.org/badge/1064030431.svg)](https://doi.org/10.5281/zenodo.17214724)

---

## **Overview**

**PhysAI** is a Python package for solving **ordinary differential equations (ODEs) and partial differential equations (PDEs)** using **Physics-Informed Neural Networks (PINNs)**. It integrates physics directly into neural network training, allowing the solution of classic physics problems without relying on traditional numerical solvers.

PhysAI supports a wide range of physics problems including:

* **ODEs:** Logistic growth, Newton’s law of cooling, damped/simple harmonic oscillators, Markov processes.
* **PDEs:** Heat equation, wave equation, Burgers' equation, KdV equation, convection-diffusion.
* **Quantum Mechanics:** Schrödinger equation.
* **Electromagnetism & Quantum Phenomena:** Planck’s law, photoelectric effect.
* **Fluid Dynamics:** 2D incompressible Navier-Stokes.
* **Static Problems:** Laplace and Poisson equations (2D/3D).

Key features:

* Flexible **PDE/ODE residual computation** for various physics laws.
* **Mixed-precision training** for faster computation on GPUs.
* Gradient clipping and learning rate schedulers supported.
* **Visualization and animations** of solutions using matplotlib.
* Weighted loss functions combining residual and boundary conditions.

---

## **Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/physai.git
cd physai
pip install -r requirements.txt
```

Python >= 3.10 recommended.

---

## **Repository Structure**

```
physai/
├── __init__.py
├── models.py          # PINN neural network class
├── trainer.py         # trainer cls for PINNs
├── visualization.py
├── pde_residual.py    # residuals for ODEs/PDEs
├── losses.py
├── utils.py 
examples/              
├── example_schrodinger.py
├── example_newton_cooling.py
├── example_markov.py
├── example_photoelectric.py
├── example_planck.py
README.md             
requirements.txt      
.gitignore             
```

---

## **Quick Start Example**

```python
import torch
from physai.models import PINN
from physai.pde_residual import pde_residual
from physai.losses import pinn_loss
from physai.visualization import plot_1d_solution

# Define a 1D logistic growth ODE
def logistic(x, y):
    r, K = 1.0, 1.0
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0] - r*y*(1 - y/K)

# Create a PINN model
model = PINN(layers=[1, 20, 20, 1], activation='tanh')

# Training points
x_train = torch.linspace(0, 5, 100).reshape(-1,1)

# Train the model
from physai.trainer import Trainer
trainer = Trainer(model, collocation_points=x_train, pde_type='logistic')
history = trainer.train(epochs=500, lr=1e-3)

# Plot solution
plot_1d_solution(model, x_train, title='Logistic Growth')
```

---

## **Training a PDE Example: 1D Heat Equation**

```python
import torch
from physai.models import PINN
from physai.trainer import Trainer
from physai.visualization import animate_2d

# PINN model
model = PINN(layers=[2, 50, 50, 1], activation='tanh')

# Collocation points
x = torch.linspace(0, 1, 50).reshape(-1,1)
t = torch.linspace(0, 2, 50).reshape(-1,1)
inputs = torch.cartesian_prod(x.squeeze(), t.squeeze())
inputs = inputs.float()

# Trainer
trainer = Trainer(model, collocation_points=inputs, pde_type='heat')
history = trainer.train(epochs=1000, lr=1e-3)

# Animate solution
animate_2d(model, x, t, title='Heat Equation Evolution')
```

---

## **Physics Problems Supported**

| Type               | Equations / Laws                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------ |
| ODE                | Logistic Growth, Simple/Damped Harmonic Oscillator, Newton's Law of Cooling, Markov Chains |
| PDE                | Heat Equation, Wave Equation, Burgers', KdV, Convection-Diffusion, Laplace, Poisson        |
| Quantum            | Schrödinger Equation, Planck's Law                                                         |
| Quantum/Electromag | Photoelectric Effect                                                                       |
| Fluid Dynamics     | 2D Incompressible Navier-Stokes                                                            |
| 3D PDEs            | Laplace 3D, Poisson 3D                                                                     |

---

## **Visualization**

* **1D plots:** `plot_1d_solution(model, x)`
* **2D surface plots:** `plot_2d_surface(model, X, Y)`
* **Animation over time:** `animate_2d(model, x, t)`
* **Training loss plots:** `plot_loss(trainer.history)`

---

## **Advanced Features**

* Mixed precision training for faster GPU computation
* Gradient clipping
* Flexible learning rate scheduling
* Weighted PINN loss for custom PDE/BC importance
* Supports custom potentials (`V(x,t)`) for Schrödinger equation

---

## Citation
If you use **PhysAI** in your research, academic publication, or official work, **citation is required**.

Please cite the software as follows:

**APA:**
> Singh, M. (2025). *PhysAI: Physics-Informed Neural Networks in PyTorch* (Version 2.5.7) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17214725

**BibTeX:**
```bibtex
@software{singh_physai_2025,
  author       = {Mankrit Singh},
  title        = {PhysAI: Physics-Informed Neural Networks in PyTorch},
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {2.5.7},
  doi          = {10.5281/zenodo.17214725},
  url          = {https://doi.org/10.5281/zenodo.17214725}
}
```

---

## **License**

Apache License. See [LICENSE](LICENSE) file.

---

**Important Notice: Ethical Use Required**  
This software **must not be used for malicious, illegal, or unethical purposes**.  
Misuse of the library voids your license under the Ethical Use Clause.


**Solve physics problems with PINNs and visualize them interactively!**
