import torch
from physai import PINN  # type: ignore
from physai.trainer import Trainer  # type: ignore
from physai.visualization import plot_1d_solution  # type: ignore

# -----------------------------
# 1. Domain: frequencies
# -----------------------------
freq = torch.linspace(0.1, 10, 200).view(-1, 1)
T_val = 2.7  # Kelvin

# -----------------------------
# 2. Exact Planck function (for reference)
# -----------------------------
def exact_planck(freq, T):
    h = 1.0
    c = 1.0
    kB = 1.0
    return (2*h*freq**3/c**2) / (torch.exp(h*freq/(kB*T)) - 1)

# -----------------------------
# 3. PINN model
# -----------------------------
model = PINN(layers=[2, 50, 50, 1])

# -----------------------------
# 4. Inputs
# -----------------------------
inputs = torch.cat([freq, torch.full_like(freq, T_val)], dim=1).float()

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model,
    collocation_points=inputs,
    pde_type="planck",
    device="cpu"
)

# -----------------------------
# 6. Train
# -----------------------------
trainer.train(
    epochs=500,
    lr=1e-3
)

# -----------------------------
# 7. Plot solution
# -----------------------------
plot_1d_solution(
    model,
    freq,
    exact=exact_planck(freq, T_val),
    title="Planck's Law"
)
