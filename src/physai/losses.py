import torch

# ------------------------------
# Basic Residual Loss
# ------------------------------
def residual_loss(residual):
    """
    Compute the mean squared residual loss for PINNs.
    Handles complex residuals using magnitude squared.
    """
    if isinstance(residual, tuple):
        # Sum of losses for multiple residuals
        return sum(torch.mean(torch.abs(r)**2) for r in residual)
    else:
        return torch.mean(torch.abs(residual)**2)

# ------------------------------
# Boundary / Initial Condition Loss
# ------------------------------
def bc_loss(pred, target):
    """
    Compute MSE loss for boundary or initial conditions.
    Handles complex by converting to complex64 if needed.
    """
    # Ensure both are complex if either is
    if pred.is_complex() or (hasattr(target, 'is_complex') and target.is_complex()):
        if not pred.is_complex():
            pred = torch.complex(pred, torch.zeros_like(pred))
        if not target.is_complex():
            target = torch.complex(target, torch.zeros_like(target))
        return torch.mean(torch.abs(pred - target)**2)
    else:
        return torch.mean((pred - target)**2)

# ------------------------------
# Combined PINN Loss
# ------------------------------
def pinn_loss(model, collocation_points, pde_type, bc_points=None, bc_values=None, **kwargs):
    """
    Compute total loss for PINNs: PDE residual + BC loss
    """
    from physai.pde_residual import pde_residual
    
    # PDE Residual
    residual = pde_residual(model, collocation_points, pde_type, **kwargs)
    res_loss = residual_loss(residual)
    
    # Boundary / Initial Condition Loss
    bc_l = torch.tensor(0.0, device=collocation_points.device)
    if bc_points is not None and bc_values is not None:
        pred_bc = model(bc_points)
        bc_l = bc_loss(pred_bc, bc_values)
    
    # Total loss (sum)
    total_loss = res_loss + bc_l
    return total_loss, res_loss, bc_l

# ------------------------------
# Weighted Loss
# ------------------------------
def weighted_pinn_loss(model, collocation_points, pde_type, bc_points=None, bc_values=None, pde_weight=1.0, bc_weight=1.0, **kwargs):
    """
    Weighted PINN loss for flexibility
    """
    total, res_loss, bc_l = pinn_loss(model, collocation_points, pde_type, bc_points, bc_values, **kwargs)
    total = pde_weight*res_loss + bc_weight*bc_l
    return total, res_loss, bc_l