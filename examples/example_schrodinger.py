import torch
from physai import PINN
from physai.trainer import Trainer
from physai.visualization import animate_2d, plot_training_loss  # Updated viz below

# -----------------------------
# 1. Domain
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_vals = torch.linspace(-5, 5, 50, device=device).view(-1, 1)  # FIXED: [50,1]
t_vals = torch.linspace(0, 2, 20, device=device).view(-1, 1)   # FIXED: [20,1]

# Collocation points [N,2]
XT = torch.cat(torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing='ij'), dim=1).reshape(-1, 2)

# -----------------------------
# 2. Potential (Harmonic oscillator)
# -----------------------------
def V(inputs):
    x = inputs[:, 0:1]
    return 0.5 * x**2  # V(x) = 0.5 x²

# -----------------------------
# 3. Initial Condition (Complex Gaussian at t=0)
# -----------------------------
bc_points = torch.cat([x_vals, torch.zeros_like(x_vals, device=device)], dim=1)  # (x, t=0)
bc_values = torch.exp(-0.5 * x_vals**2).to(torch.complex64)  # Real Gaussian ψ(x,0), imag=0

# Test non-zero BC
print(f"BC sample |ψ(0,0)|: {torch.abs(bc_values[torch.argmax(x_vals.squeeze() == 0)]).item():.3f} (should be ~1.0)")

# -----------------------------
# 4. PINN Model (Complex output)
# -----------------------------
model = PINN(layers=[2, 50, 50, 50, 1], complex_output=True, device=device)  # Last=1 doubles to 2 internally

# Test output
test_out = model(XT[:5])
print(f"Model test: Shape={test_out.shape}, Complex={test_out.is_complex()} (should be [5], True)")

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    collocation_points=XT,
    bc_points=bc_points,
    bc_values=bc_values,
    pde_type="schrodinger",
    device=device
)

# -----------------------------
# 6. Train (with BC weight for better IC fit)
# -----------------------------
history = trainer.train(
    epochs=500,
    lr=1e-3,
    V=V,  # Potential
    hbar=1.0,
    m=1.0,
    clip_grad=1.0,
    verbose=True,
    bc_weight=10.0  # FIXED: Weight BC higher (helps plateau drop to <1e-2)
)

# -----------------------------
# 7. Visualize
# -----------------------------
trainer.plot_training_loss()  # Decreasing curve (Res drops, BC to ~0)

# FIXED: [N,1] inputs – no crash
animate_2d(model, x_vals, t_vals, title="Schrödinger |ψ(x,t)| Evolution", interval=200, device=device)