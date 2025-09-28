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
        # self.use_amp = self.device.startswith("cuda") and pde_type != "schrodinger"
        self.use_amp = False
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
            optimizer.zero_grad(set_to_none=True)  # FIXED: Efficient zero_grad (reduces GPU memory)
            # Use autocast only if enabled
            ctx = autocast(device_type=self.device, enabled=self.use_amp)
            with ctx:
                total, res_l, bc_l = pinn_loss(
                    self.model, x, self.pde_type, bc_points=bc_x, bc_values=bc_y, **kwargs
                )

            # FIXED: Ensure total is scalar for backward (safety, handles non-scalar rare cases)
            if total.dim() > 0:
                total = total.mean()
            # If complex (rare), take real part for scalar backward
            if hasattr(total, 'is_complex') and total.is_complex():
                total = total.real

            if self.use_amp:
                # Scale and backward
                scaled_loss = self.scaler.scale(total)
                scaled_loss.backward()

                # FIXED: Always unscale before step (required for inf checks, even without clipping)
                self.scaler.unscale_(optimizer)

                # Optional clipping (after unscale)
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

                # Step and update (now with inf checks recorded)
                try:  # FIXED: Optional fallback if AMP re-enabled and assertion hits
                    self.scaler.step(optimizer)
                    self.scaler.update()
                except AssertionError as e:
                    if "No inf checks" in str(e):
                        print(f"Epoch {epoch+1}: AMP fallback to FP32 (inf checks missing)")
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    p.grad.data.div_(self.scaler.get_scale())
                        optimizer.step()
                        self.scaler.update()
                    else:
                        raise e
            else:
                total.backward()

                # Optional clipping (FP32)
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

                optimizer.step()

            if scheduler:
                scheduler.step()

            # Detach for logging (avoid graph buildup)
            self.history["total_loss"].append(total.detach().item())
            self.history["res_loss"].append(res_l.detach().item())
            self.history["bc_loss"].append(bc_l.detach().item())

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} | Total: {total.detach().item():.6e} | "
                      f"Res: {res_l.detach().item():.6e} | BC: {bc_l.detach().item():.6e}")

        return self.history

    def plot_training_loss(self):
        """Plot training loss curves"""
        plot_loss(self.history, title=f"Training Loss for {self.pde_type}")