import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # FIXED: Correct import (capital S, no 'grad_scaler')
import warnings
try:
    warnings.filterwarnings('ignore', category=torch.exceptions.ComplexHalfWarning)
except AttributeError:
    # Ignore if the warning class doesn't exist (older PyTorch)
    pass

class PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) with advanced optimization features."""

    def __init__(self, layers, activation='tanh', device=None, complex_output=False):
        super().__init__()
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.complex_output = complex_output
        # Adjust last layer size if complex output is desired
        if self.complex_output:
            layers[-1] = layers[-1] * 2 # Output real and imaginary parts
        # Build the neural network
        self.model = self._build_network(layers, activation).to(self.device)
        # Store training history
        self.history = {"loss": []}

    def _build_network(self, layers, activation):
        """Construct a fully connected network with specified activation and Xavier initialization."""
        net = []
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        act = activations.get(activation.lower(), nn.Tanh())

        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            net.append(linear)
            if i < len(layers) - 2:
                net.append(act)
        return nn.Sequential(*net)

    def forward(self, x):
        # Ensure x is on device
        x = x.to(self.device)
        output = self.model(x)
        if self.complex_output:
            # FIXED: Robust check - handle if already complex or wrong shape
            if hasattr(output, 'is_complex') and output.is_complex():
                return output  # Already complex, return as-is
            if output.shape[-1] == 2:
                real = output[..., 0]
                imag = output[..., 1]
            else:
                raise ValueError(f"Expected output shape with last dim 2 for complex, got {output.shape}")
            
            # MODIFIED: Removed dtype from torch.complex for broader compatibility
            complex_output = torch.complex(real, imag)  # PyTorch infers dtype from inputs
            complex_output = complex_output.to(torch.complex64)  # optional, ensure 32-bit
            return complex_output
        return output

    def physics_loss(self, x, physics_fn):
        """Compute physics-integrated loss."""
        x = x.to(self.device).requires_grad_(True)
        y = self.forward(x)
        residual = physics_fn(x, y)
        # FIXED: Ensure scalar (mean if non-scalar)
        if residual.dim() > 0:
            residual = residual.mean()
        return torch.mean(torch.abs(residual) ** 2) # Use abs for complex residuals

    def train_model(self, x, physics_fn, lr=1e-3, epochs=1000, verbose=True,
                    clip_grad=None, scheduler=None, use_amp=True):
        """Train the PINN with advanced options: AMP, gradient clipping, scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = GradScaler(enabled=use_amp)

        if scheduler:
            scheduler = scheduler(optimizer)

        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)  # FIXED: Efficient zero_grad

            with autocast(device_type=self.device, enabled=use_amp):  # FIXED: Add device_type
                loss = self.physics_loss(x, physics_fn)
                # FIXED: Ensure scalar loss
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()

            # FIXED: Always unscale before step/clip (prevents assertion if use_amp=True)
            scaler.unscale_(optimizer)

            # Optional clipping (after unscale)
            if clip_grad:
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

            # FIXED: Step with fallback
            try:
                scaler.step(optimizer)
                scaler.update()
            except AssertionError as e:
                if "No inf checks" in str(e):
                    # Fallback: Manual step
                    for p in self.parameters():
                        if p.grad is not None:
                            p.grad.data.div_(scaler.get_scale())
                    optimizer.step()
                    scaler.update()
                else:
                    raise e

            if scheduler:
                scheduler.step()

            self.history["loss"].append(loss.detach().item())  # FIXED: Detach for safety
            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.detach().item():.6f}")

        return self.history