import torch
from torch import optim
from .losses import pinn_loss
from .visualization import plot_loss

class Trainer:
    """
    Trainer class for Physics-Informed Neural Networks (PINNs)
    Supports:
        - Mixed precision training (AMP)
        - Gradient clipping
        - Optional learning rate scheduler
        - Logging & plotting
        - Flexible handling of ODE/PDE types
    """

    def __init__(self, model, collocation_points, pde_type, bc_points=None, bc_values=None, device=None):
        self.model = model
        self.x = collocation_points
        self.pde_type = pde_type
        self.bc_x = bc_points
        self.bc_y = bc_values
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.history = {"total_loss": [], "res_loss": [], "bc_loss": []}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.startswith("cuda"))

    def train(self, epochs=1000, lr=1e-3, scheduler_fn=None, clip_grad=None, verbose=True, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        x = self.x.to(self.device)
        bc_x = self.bc_x.to(self.device) if self.bc_x is not None else None
        bc_y = self.bc_y.to(self.device) if self.bc_y is not None else None

        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                total, res_l, bc_l = pinn_loss(
                    self.model, x, self.pde_type, bc_points=bc_x, bc_values=bc_y, **kwargs
                )

            self.scaler.scale(total).backward()

            if clip_grad:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            self.scaler.step(optimizer)
            self.scaler.update()

            if scheduler:
                scheduler.step()

            self.history["total_loss"].append(total.item())
            self.history["res_loss"].append(res_l.item())
            self.history["bc_loss"].append(bc_l.item())

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} | Total: {total.item():.6e} | "
                      f"Res: {res_l.item():.6e} | BC: {bc_l.item():.6e}")

        return self.history

    def plot_training_loss(self):
        """Plot training loss curves"""
        plot_loss(self.history, title=f"Training Loss for {self.pde_type}")
