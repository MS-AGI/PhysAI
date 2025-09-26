import torch
from physai import PINN  # type: ignore
from physai.trainer import Trainer  # type: ignore
from physai.visualization import animate_2d  # type: ignore

# -----------------------------
# 1. Domain
# -----------------------------
x_vals = torch.linspace(-5, 5, 200)
t_vals = torch.linspace(0, 2, 50)

# Create collocation points manually (list comprehension)
XT = torch.tensor([[xi.item(), ti.item()] for ti in t_vals for xi in x_vals],
                  dtype=torch.float32,
                  requires_grad=True)  # critical for autograd

# -----------------------------
# 2. PINN model
# -----------------------------
model = PINN(layers=[2, 50, 50, 1])

# -----------------------------
# 3. Trainer setup
# -----------------------------
trainer = Trainer(
    model,
    collocation_points=XT,
    pde_type="markov",
    device="cpu"
)

# -----------------------------
# 4. Train
# -----------------------------
trainer.train(
    epochs=500,
    lr=1e-3,
    D=0.1  # diffusion coefficient
)

# -----------------------------
# 5. Animate diffusion
# -----------------------------
animate_2d(model, x_vals, t_vals, title="Markov Diffusion / Fokker-Planck")
