import torch
from physai import PINN
from physai.trainer import Trainer
from physai.visualization import plot_1d_solution

# 1. Domain
freq = torch.linspace(0.1, 10, 200).view(-1, 1)
T_val = 2.7
inputs = torch.cat([freq, torch.full_like(freq, T_val)], dim=1).float()
inputs.requires_grad_(True)

# 2. Exact Planck function
def exact_planck(freq_tensor, T_tensor):
    h = 1.0; c = 1.0; kB = 1.0
    exp_term = torch.exp(h * freq_tensor / (kB * T_tensor))
    return (2 * h * freq_tensor**3 / c**2) / (exp_term - 1 + 1e-9)

# 3. PINN
model = PINN(layers=[2, 50, 50, 1])

# 4. Device
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = inputs.to(device)
model.to(device)

# 5. Trainer
trainer = Trainer(model, collocation_points=inputs, pde_type="planck", device=device)

# 6. Train
trainer.train(
    epochs=500,
    lr=1e-3,
    exact_planck=lambda f, T: exact_planck(f, T)  # Use T dynamically
)

# 7. Plot
plot_1d_solution(
    model,
    freq,
    exact=exact_planck(freq, torch.full_like(freq, T_val)),
    title="Planck's Law"
)
