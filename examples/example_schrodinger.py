import torch  
from physai import PINN
from physai.trainer import Trainer
from physai.visualization import animate_2d

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
# Initial condition for Schrödinger is typically complex.
# Here, we assume a purely real Gaussian initial wavefunction.
bc_values = torch.exp(-0.5 * x0**2).to(torch.complex64) # Ensure bc_values are complex

# -----------------------------
# 4. PINN model
# -----------------------------
# Set complex_output=True for Schrödinger equation
model = PINN(layers=[2, 50, 50, 50, 1], complex_output=True)

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model,
    collocation_points=XT,
    bc_points=bc_points,
    bc_values=bc_values,
    pde_type="schrodinger",
 # Removed explicit device setting to allow GPU if available
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
    # batch_size=500  # Removed ignored argument
)

# -----------------------------
# 7. Visualization
# -----------------------------
animate_2d(model, x_vals, t_vals, title="Schrödinger Equation Evolution (Magnitude)")