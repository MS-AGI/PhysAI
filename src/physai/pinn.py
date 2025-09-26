import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.amp import grad_scaler as GradScaler
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
            # Handle shape: assume output is (..., 2) for [real, imag]
            if output.shape[-1] == 2:
                real = output[..., 0]
                imag = output[..., 1]
            else:
                raise ValueError(f"Expected output shape with last dim 2 for complex, got {output.shape}")
            return torch.complex(real, imag, dtype=torch.complex64)
        return output

    def physics_loss(self, x, physics_fn):
        """Compute physics-integrated loss."""
        x = x.to(self.device).requires_grad_(True)
        y = self.forward(x)
        residual = physics_fn(x, y)
        return torch.mean(torch.abs(residual) ** 2) # Use abs for complex residuals

    def train_model(self, x, physics_fn, lr=1e-3, epochs=1000, verbose=True,
                    clip_grad=None, scheduler=None, use_amp=True):
        """Train the PINN with advanced options: AMP, gradient clipping, scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = GradScaler(enabled=use_amp)

        if scheduler:
            scheduler = scheduler(optimizer)

        for epoch in range(epochs):
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss = self.physics_loss(x, physics_fn)

            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            self.history["loss"].append(loss.item())
            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

        return self.history