import torch
from torch import optim
from torch.amp import autocast, GradScaler
from .losses import pinn_loss
from .visualization import plot_loss

class Trainer:
    """
    Trainer class for Physics-Informed Neural Networks (PINNs)
    Supports mixed precision, but disables for complex PDEs.
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
        # Disable AMP for complex PDEs (e.g., schrodinger) due to PyTorch limitations
        self.use_amp = self.device.startswith("cuda") and pde_type != "schrodinger"
        self.scaler = GradScaler(enabled=self.use_amp)

    def train(self, epochs=1000, lr=1e-3, scheduler_fn=None, clip_grad=None, verbose=True, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        # -----------------------------
        # Ensure inputs require gradients WITHOUT detaching (preserve graph)
        # -----------------------------
        def make_grad_tensor(t):
            if t is None:
                return None
            # Remove .detach() to avoid breaking grad_fn
            return t.clone().to(self.device).requires_grad_(True)

        x = make_grad_tensor(self.x)
        bc_x = make_grad_tensor(self.bc_x)
        
        # Ensure bc_y is on the correct device and dtype
        if self.bc_y is not None:
            bc_y = self.bc_y.to(self.device)
            # For complex models, ensure bc_y is complex
            if hasattr(self.model, 'complex_output') and self.model.complex_output and not bc_y.is_complex():
                bc_y = torch.complex(bc_y, torch.zeros_like(bc_y))
        else:
            bc_y = None

        # -----------------------------
        # Training loop
        # -----------------------------
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Use autocast only if enabled
            ctx = autocast(device_type=self.device, enabled=self.use_amp)
            with ctx:
                total, res_l, bc_l = pinn_loss(
                    self.model, x, self.pde_type, bc_points=bc_x, bc_values=bc_y, **kwargs
                )

            if self.use_amp:
                self.scaler.scale(total).backward()
                if clip_grad:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                optimizer.step()

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