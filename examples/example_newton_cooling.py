import torch
from physai import PINN  # type: ignore
from physai.trainer import Trainer  # type: ignore
from physai.visualization import plot_1d_solution  # type: ignore

# -----------------------------
# 1. Domain
# -----------------------------
t = torch.linspace(0, 10, 100).view(-1, 1).float()

# -----------------------------
# 2. PINN model
# -----------------------------
model = PINN(layers=[1, 20, 20, 1])

# -----------------------------
# 3. Newton cooling parameters
# -----------------------------
T_env = 25.0
k = 0.2
T0 = torch.tensor([[90.0]], dtype=torch.float32)

# -----------------------------
# 4. Boundary points
# -----------------------------
bc_points = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
bc_values = T0

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model,
    collocation_points=t,
    pde_type="newton_cooling",
    bc_points=bc_points,
    bc_values=bc_values,
    device="cpu"
)

# -----------------------------
# 6. Train the model
# -----------------------------
trainer.train(
    epochs=300,
    lr=1e-3,
    k=k,
    T_env=T_env
)

# -----------------------------
# 7. Visualize
# -----------------------------
plot_1d_solution(model, t, title="Newton's Law of Cooling")
