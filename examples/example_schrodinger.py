import torch
from physai.pde_residual import pde_residual #type: ignore
from physai.losses import pinn_loss #type: ignore
from physai.trainer import Trainer #type: ignore
from physai.visualization import plot_1d_solution, animate_2d #type: ignore
from physai.models import PINN #type: ignore

# Define domain
x = torch.linspace(-5, 5, 200).view(-1,1)
t = torch.linspace(0, 2, 50).view(-1,1)
XT = torch.cartesian_prod(x.squeeze(), t.squeeze())

# Define potential (harmonic oscillator)
def V(inputs):
    x = inputs[:,0:1]
    return 0.5 * x**2

# PINN model
model = PINN(layers=[2, 50, 50, 50, 1])

# Trainer
trainer = Trainer(model, XT, pde_type="schrodinger", device="cpu")
history = trainer.train(epochs=500, lr=1e-3, V=V)

# Visualize solution over time
animate_2d(model, x, t, title="Schrödinger Equation Evolution")
