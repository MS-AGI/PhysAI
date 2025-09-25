import torch
from physai.models import PINN #type: ignore
from physai.trainer import Trainer #type: ignore
from physai.visualization import animate_2d #type: ignore

# Domain
x = torch.linspace(-5,5,200).view(-1,1)
t = torch.linspace(0,2,50).view(-1,1)
XT = torch.cartesian_prod(x.squeeze(), t.squeeze())

# PINN model
model = PINN(layers=[2, 50, 50, 1])

# Trainer
trainer = Trainer(model, XT, pde_type="markov", device="cpu")
trainer.train(epochs=500, lr=1e-3, D=0.1)

# Animate diffusion
animate_2d(model, x, t, title="Markov Diffusion / Fokker-Planck")
