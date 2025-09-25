import os

# ------------------------------
# Config
# ------------------------------
LIB_NAME = "physai"
SRC_DIR = f"src/{LIB_NAME}"
EXAMPLES_DIR = "examples"
TESTS_DIR = "tests"

# ------------------------------
# Folder structure
# ------------------------------
folders = [
    SRC_DIR,
    EXAMPLES_DIR,
    TESTS_DIR
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ------------------------------
# __init__.py
# ------------------------------
init_code = """from .pinn import PINN
from .physics import *
from .losses import *
from .trainer import *
from .visualization import *
from .utils import *
"""
with open(os.path.join(SRC_DIR, "__init__.py"), "w") as f:
    f.write(init_code)

# ------------------------------
# pinn.py
# ------------------------------
pinn_code = """import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class PINN(nn.Module):
    \"\"\"Physics-Informed Neural Network (PINN) with AMP, gradient clipping, scheduler.\"\"\"

    def __init__(self, layers, activation='tanh', device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_network(layers, activation).to(self.device)
        self.history = {"loss": []}

    def _build_network(self, layers, activation):
        net = []
        activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU()}
        act = activations.get(activation.lower(), nn.Tanh())
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            net.append(linear)
            if i < len(layers)-2:
                net.append(act)
        return nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

    def physics_loss(self, x, physics_fn):
        x = x.to(self.device).requires_grad_(True)
        y = self.forward(x)
        residual = physics_fn(x, y)
        return torch.mean(residual**2)

    def train_model(self, x, physics_fn, lr=1e-3, epochs=1000, verbose=True,
                    clip_grad=None, scheduler=None, use_amp=True):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = GradScaler(enabled=use_amp)
        if scheduler:
            scheduler = scheduler(optimizer)

        for epoch in range(epochs):
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss = self.physics_loss(x, physics_fn)
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            self.history["loss"].append(loss.item())
            if verbose and (epoch % max(epochs//10,1) == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
        return self.history
"""

with open(os.path.join(SRC_DIR, "pinn.py"), "w") as f:
    f.write(pinn_code)

# ------------------------------
# physics.py
# ------------------------------
physics_code = """import torch

def derivative(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return y

# 1D ODEs
def dy_dx_equals_y(x, model):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - y

def dy_dx_equals_func(x, model, func):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - func(x)

def second_order_ode(x, model):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x, 2) + y

def damped_harmonic_oscillator(x, model, gamma=0.1):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x, 2) + gamma*derivative(y, x) + y

def logistic_growth(x, model, r=1.0, K=1.0):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - r*y*(1 - y/K)

# 1D PDEs
def heat_equation(u, x, t, model):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) - derivative(u_val, x, 2)

def wave_equation(u, x, t, model, c=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t, 2) - c**2 * derivative(u_val, x, 2)

def burgers_equation(u, x, t, model, nu=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + u_val*derivative(u_val, x) - nu*derivative(u_val, x, 2)

def kdv_equation(u, x, t, model, alpha=6.0, beta=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + alpha*u_val*derivative(u_val, x) + beta*derivative(u_val, x, 3)

def convection_diffusion(u, x, t, model, c=1.0, D=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + c*derivative(u_val, x) - D*derivative(u_val, x, 2)

# 2D PDEs
def laplace_equation(u, x, y, model):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u_val = model(torch.cat([x, y], dim=1))
    return derivative(u_val, x, 2) + derivative(u_val, y, 2)

def poisson_equation(u, x, y, f, model):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u_val = model(torch.cat([x, y], dim=1))
    return derivative(u_val, x, 2) + derivative(u_val, y, 2) - f(x, y)
"""

with open(os.path.join(SRC_DIR, "physics.py"), "w") as f:
    f.write(physics_code)

# ------------------------------
# losses.py
# ------------------------------
losses_code = """def pde_loss(residual):
    return residual

def ode_loss(residual):
    return residual
"""

with open(os.path.join(SRC_DIR, "losses.py"), "w") as f:
    f.write(losses_code)

# ------------------------------
# trainer.py
# ------------------------------
trainer_code = """def train(model, x, physics_fn, **kwargs):
    return model.train_model(x, physics_fn, **kwargs)
"""

with open(os.path.join(SRC_DIR, "trainer.py"), "w") as f:
    f.write(trainer_code)

# ------------------------------
# visualization.py
# ------------------------------
viz_code = """import matplotlib.pyplot as plt

def plot_loss(history):
    plt.figure(figsize=(6,4))
    plt.plot(history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()
"""

with open(os.path.join(SRC_DIR, "visualization.py"), "w") as f:
    f.write(viz_code)

# ------------------------------
# utils.py
# ------------------------------
utils_code = """import torch

def to_device(tensor, device='cpu'):
    return tensor.to(device)
"""

with open(os.path.join(SRC_DIR, "utils.py"), "w") as f:
    f.write(utils_code)

# ------------------------------
# Example script
# ------------------------------
example_code = """import torch
from physai.pinn import PINN
from physai.physics import dy_dx_equals_y
from physai.visualization import plot_loss

x = torch.linspace(0, 2, 50).reshape(-1,1)
model = PINN([1,50,50,1], activation='tanh')
history = model.train_model(x, lambda x,y: dy_dx_equals_y(x, model), lr=1e-3, epochs=300)
plot_loss(history)
"""

with open(os.path.join(EXAMPLES_DIR, "ode_example.py"), "w") as f:
    f.write(example_code)

# ------------------------------
# Test script
# ------------------------------
test_code = """import torch
from physai.pinn import PINN
from physai.physics import dy_dx_equals_y

def test_loss_nonnegative():
    x = torch.tensor([[0.0], [1.0]], requires_grad=True)
    model = PINN([1,10,1])
    loss = model.physics_loss(x, lambda x,y: dy_dx_equals_y(x, model))
    assert loss.item() >= 0

if __name__ == '__main__':
    test_loss_nonnegative()
    print('Test passed!')
"""

with open(os.path.join(TESTS_DIR, "test_pinn.py"), "w") as f:
    f.write(test_code)

# ------------------------------
# PyPI Boilerplate
# ------------------------------
pyproject_code = """[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "physai"
version = "0.1.0"
description = "Advanced Physics-Informed Neural Networks (PINNs) library"
readme = "README.md"
authors = [{ name="Mankrit Singh", email="your_email@example.com" }]
license = { text="MIT" }
dependencies = ["torch>=2.0", "matplotlib>=3.0"]

[project.urls]
Homepage = "https://github.com/yourusername/physai"
"""

with open("pyproject.toml", "w") as f:
    f.write(pyproject_code)

setup_cfg_code = """[metadata]
license_files = LICENSE

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.8

[options.packages.find]
where = src
"""

with open("setup.cfg", "w") as f:
    f.write(setup_cfg_code)

print("PhysAI library structure generated successfully!")
