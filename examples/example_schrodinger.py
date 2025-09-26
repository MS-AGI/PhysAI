import torch
from physai import PINN  # type: ignore
from physai.trainer import Trainer  # type: ignore
from physai.visualization import animate_2d  # type: ignore

# -----------------------------
# 1. Domain
# -----------------------------
x_vals = torch.linspace(-5, 5, 50)
t_vals = torch.linspace(0, 2, 20)

# Create collocation points manually (list comprehension)
XT = torch.tensor([[xi.item(), ti.item()] for ti in t_vals for xi in x_vals], 
                  dtype=torch.float32, requires_grad=True)

# -----------------------------
# 2. Potential
# -----------------------------
def V(inputs):
    x = inputs[:,0:1]
    return 0.5 * x**2

# -----------------------------
# 3. Initial condition (Gaussian)
# -----------------------------
x0 = x_vals.view(-1,1)
t0 = torch.zeros_like(x0)
bc_points = torch.cat([x0, t0], dim=1).float().requires_grad_(True)
bc_values = torch.exp(-0.5 * x0**2)  # Gaussian

# -----------------------------
# 4. PINN model
# -----------------------------
model = PINN(layers=[2, 50, 50, 50, 1])

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model,
    collocation_points=XT,
    bc_points=bc_points,
    bc_values=bc_values,
    pde_type="schrodinger",
    device="cpu"
)

# -----------------------------
# 6. Train
# -----------------------------
history = trainer.train(
    epochs=500,
    lr=1e-3,
    V=V,
    hbar=1.0,
    m=1.0,
    batch_size=500  # reduce batch size for memory
)

# -----------------------------
# 7. Visualization
# -----------------------------
animate_2d(model, x_vals, t_vals, title="Schrödinger Equation Evolution")
