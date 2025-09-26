import torch
from physai import PINN
from physai.trainer import Trainer
from physai.visualization import plot_1d_solution

# -----------------------------
# 1. Domain: frequencies
# -----------------------------
freq = torch.linspace(0.1, 10, 200).view(-1, 1)
T_val = 2.7  # Kelvin

# -----------------------------
# 2. Exact Planck function (for reference)
# -----------------------------
def exact_planck(freq_tensor, T_scalar):
    h = 1.0
    c = 1.0
    kB = 1.0
    # Ensure operations are on tensors and handle potential division by zero for exp(0)-1
    exp_term = torch.exp(h * freq_tensor / (kB * T_scalar))
    # Add a small epsilon to prevent division by zero if exp_term happens to be 1
    return (2 * h * freq_tensor**3 / c**2) / (exp_term - 1 + 1e-9)

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
    # device="cpu" # Removed explicit device setting to allow GPU if available
)

# -----------------------------
# 6. Train
# -----------------------------
trainer.train(
    epochs=500,
    lr=1e-3,
    exact_planck=lambda f, T: exact_planck(f, T_val) # Pass the exact Planck function with fixed T_val
)

# -----------------------------
# 7. Plot solution
# -----------------------------
plot_1d_solution(
    model,
    freq, # Pass freq for x-axis, model will take inputs
    exact=exact_planck(freq, T_val),
    title="Planck's Law"
)
