import torch
from physai import PINN  # Your updated pinn.py
from physai.trainer import Trainer  # Your updated trainer.py
from physai.visualization import animate_2d, plot_training_loss  # Updated viz below

# -----------------------------
# 1. Domain (1D space + time)
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_vals = torch.linspace(-5, 5, 50, device=device).view(-1, 1)  # FIXED: [50,1] for model/animation
t_vals = torch.linspace(0, 2, 20, device=device).view(-1, 1)   # FIXED: [20,1]

# Collocation points [N,2] (x,t mesh) – FIXED: Efficient meshgrid (no xx/tt)
XT = torch.cat(torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing='ij'), dim=1).reshape(-1, 2)

# -----------------------------
# 2. Initial Condition (Non-Trivial Gaussian BC at t=0)
# -----------------------------
sigma = 1.0  # Gaussian width
bc_points = torch.cat([x_vals, torch.zeros_like(x_vals, device=device)], dim=1)  # [50,2]: (x, t=0)
bc_values = torch.exp(-x_vals**2 / (2 * sigma**2)).squeeze(-1)  # [50] real Gaussian (peaks ~1 at x=0)

# Test: Non-zero BC
print(f"BC sample: x=0 value={bc_values[torch.argmin(torch.abs(x_vals.squeeze()))].item():.3f} (should be ~1.0)")
print(f"BC mean: {bc_values.mean().item():.3f} (should be ~0.4, non-zero)")

# -----------------------------
# 3. PINN Model (Real output for Markov diffusion)
# -----------------------------
model = PINN(layers=[2, 50, 50, 50, 1], activation='tanh', device=device)  # Last=1 for real u(x,t)

# Test non-zero output (post-Xavier)
test_in = torch.tensor([[0.0, 0.0]], device=device, requires_grad=True)
test_out = model(test_in)
print(f"Model test output at (0,0): {test_out.item():.6f} (should be non-zero, e.g., 0.01-0.1)")

# -----------------------------
# 4. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    collocation_points=XT,
    pde_type="markov",
    bc_points=bc_points,  # FIXED: Pass non-trivial BC
    bc_values=bc_values,  # FIXED: Gaussian values
    device=device
)

# -----------------------------
# 5. Train
# -----------------------------
history = trainer.train(
    epochs=500,
    lr=1e-3,
    D=0.1,  # Diffusion coefficient
    clip_grad=1.0,  # Stability
    verbose=True
)

# -----------------------------
# 6. Visualize
# -----------------------------
trainer.plot_training_loss()  # FIXED: Should show decreasing curve (not flat zero)

# FIXED: x_vals/t_vals already [N,1] – animation works
animate_2d(model, x_vals, t_vals, title="Markov Diffusion (Gaussian Spreading)", interval=200, device=device)
