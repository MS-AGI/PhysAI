import torch
from physai import PINN #type: ignore
from physai.trainer import Trainer #type: ignore
from physai.visualization import plot_1d_solution #type: ignore

# Domain: photon frequencies
freq = torch.linspace(0, 10, 100).view(-1,1)
work_func = 5.0

# PINN model
model = PINN(layers=[1, 20, 20, 1])

# Trainer
trainer = Trainer(model, freq, pde_type="photoelectric")
trainer.train(epochs=200, lr=1e-3, work_func=work_func)

# Visualize
plot_1d_solution(model, freq, title="Photoelectric Effect")
