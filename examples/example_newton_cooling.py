import torch
from physai.models import PINN #type: ignore
from physai.trainer import Trainer #type: ignore
from physai.visualization import plot_1d_solution #type: ignore

# Domain
t = torch.linspace(0, 10, 100).view(-1,1)

# PINN model
model = PINN(layers=[1, 20, 20, 1])

# Newton cooling parameters
T_env = 25.0
k = 0.2
T0 = torch.tensor([[90.0]])

# Boundary points
bc_points = torch.tensor([[0.0]])
bc_values = T0

# Trainer
trainer = Trainer(model, t, pde_type="newton_cooling", bc_points=bc_points, bc_values=bc_values)
trainer.train(epochs=300, lr=1e-3, k=k, T_env=T_env)

# Visualize
plot_1d_solution(model, t, title="Newton's Law of Cooling")
