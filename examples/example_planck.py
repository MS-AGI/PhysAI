import torch
from physai.models import PINN #type: ignore
from physai.trainer import Trainer #type: ignore
from physai.visualization import plot_1d_solution #type: ignore

# Domain: frequencies
freq = torch.linspace(0.1, 10, 200).view(-1,1)
T = 2.7  # Kelvin

# Exact Planck function
def exact_planck(freq, T):
    h = 1.0; c = 1.0; kB = 1.0
    return (2*h*freq**3/c**2) / (torch.exp(h*freq/(kB*T)) - 1)

# PINN model
model = PINN(layers=[2, 50, 50, 1])

# Create inputs
inputs = torch.cat([freq, torch.full_like(freq, T)], dim=1)

# Trainer
trainer = Trainer(model, inputs, pde_type="planck")
trainer.train(epochs=500, lr=1e-3, exact_planck=exact_planck)

# Plot
plot_1d_solution(model, freq, exact=exact_planck(freq, T), title="Planck's Law")
