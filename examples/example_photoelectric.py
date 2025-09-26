import torch
from physai import PINN  # type: ignore
from physai.trainer import Trainer  # type: ignore
from physai.visualization import plot_1d_solution  # type: ignore

# ----------------------------
# 1. Define domain & constants
# ----------------------------
work_func = 5.0  # φ (threshold frequency)
freq = torch.linspace(0, 10, 200).view(-1, 1)  # photon frequencies

# Optional normalization for stable training
freq_mean, freq_std = freq.mean(), freq.std()
freq_norm = (freq - freq_mean) / freq_std

# Boundary condition: at ν = φ, kinetic energy = 0
bc_points = torch.tensor([[work_func]])  # input where BC is applied
bc_values = torch.tensor([[0.0]])        # value at that point

# ----------------------------
# 2. Initialize PINN
# ----------------------------
model = PINN(layers=[1, 64, 64, 64, 1])  # deeper network for accuracy

# ----------------------------
# 3. Train the PINN
# ----------------------------
trainer = Trainer(
    model=model,
    collocation_points=freq_norm,
    pde_type="photoelectric",
    bc_points=bc_points,
    bc_values=bc_values
)

# Train for more epochs
trainer.train(epochs=5000, lr=1e-3, work_func=work_func)

# ----------------------------
# 4. Visualize the result
# ----------------------------
# Predict on unnormalized domain for visualization
plot_1d_solution(model, freq_norm, title="Photoelectric Effect (PINN Prediction)")
