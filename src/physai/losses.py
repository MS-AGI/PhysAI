import torch

# ------------------------------
# Basic Residual Loss
# ------------------------------
def residual_loss(residual):
    """
    Compute the mean squared residual loss for PINNs.
    residual: tensor or tuple of tensors (for systems like Navier-Stokes)
    """
    if isinstance(residual, tuple):
        # Sum of losses for multiple residuals
        return sum(torch.mean(r**2) for r in residual)
    else:
        return torch.mean(residual**2)

# ------------------------------
# Boundary / Initial Condition Loss
# ------------------------------
def bc_loss(pred, target):
    """
    Compute MSE loss for boundary or initial conditions.
    pred: predicted values at BC/IC points
    target: true values at BC/IC points
    """
    return torch.mean((pred - target)**2)

# ------------------------------
# Combined PINN Loss
# ------------------------------
def pinn_loss(model, collocation_points, pde_type, bc_points=None, bc_values=None, **kwargs):
    """
    Compute total loss for PINNs: PDE residual + BC loss
    model: PINN model
    collocation_points: points where PDE is enforced
    pde_type: string specifying PDE/ODE type
    bc_points: tensor for boundary/initial points
    bc_values: tensor for boundary/initial values
    kwargs: extra parameters for PDE residual (nu, r, K, gamma, etc.)
    """
    from physai import pde_residual
    
    # PDE Residual
    residual = pde_residual(model, collocation_points, pde_type, **kwargs)
    res_loss = residual_loss(residual)
    
    # Boundary / Initial Condition Loss
    bc_l = torch.tensor(0.0)
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
