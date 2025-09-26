import torch
from physai import PINN
from physai.trainer import Trainer
from physai.visualization import plot_1d_solution

# ----------------------------
# 1. Define domain & constants
# ----------------------------
work_func = 5.0  # φ (threshold frequency)
freq = torch.linspace(0, 10, 200).view(-1, 1)  # photon frequencies

# Optional normalization for stable training
freq_mean, freq_std = freq.mean(), freq.std()
freq_norm = (freq - freq_mean) / freq_std

# Normalize work_func for consistency with freq_norm
normalized_work_func = (work_func - freq_mean) / freq_std

# Boundary condition: at ν = φ, kinetic energy = 0
# Normalize bc_points for consistency
bc_points_original = torch.tensor([[work_func]])
bc_points_norm = (bc_points_original - freq_mean) / freq_std
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
    bc_points=bc_points_norm, # Use normalized bc_points
    bc_values=bc_values
    # device="cpu" # Removed explicit device setting to allow GPU if available
)

# Train for more epochs, passing the normalized work_func
trainer.train(epochs=5000, lr=1e-3, work_func=normalized_work_func)

# ----------------------------
# 4. Visualize the result
# ----------------------------
# Predict on unnormalized domain for visualization
# For plotting, we want to see the output corresponding to the original frequency range.
# The model was trained on freq_norm, so we pass freq_norm to the model.
# The plot_1d_solution function will handle the x-axis correctly.
plot_1d_solution(model, freq_norm, title="Photoelectric Effect (PINN Prediction)")
