import torch

# ------------------------------
# Utility: derivatives (ROBUST HIGHER-ORDER FIX)
# ------------------------------
def derivative(y, x, order=1):
    # Safety checks
    if not y.requires_grad:
        raise RuntimeError(f"Cannot compute derivative of y (shape {y.shape}, requires_grad={y.requires_grad}). "
                          f"Ensure inputs to model have requires_grad=True and no detach() in pipeline.")
    if not x.requires_grad:
        raise RuntimeError(f"Cannot differentiate w.r.t. x (shape {x.shape}, requires_grad={x.requires_grad}). "
                          f"Set x.requires_grad_(True) explicitly for slices.")
    
    current_y = y  # Start with y
    for i in range(order):
        # Compute gradient
        grad_outputs = torch.ones_like(current_y, dtype=current_y.dtype, requires_grad=False)
        grad = torch.autograd.grad(
            outputs=current_y,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,  # FIXED: Retain for higher-order stability
            allow_unused=True
        )[0]

        # FIXED: Handle None grad (e.g., unused paths) with zeros that support further grads
        if grad is None:
            grad = torch.zeros_like(current_y, dtype=current_y.dtype, requires_grad=True)

        # CRITICAL FIX: Explicitly enable requires_grad on grad for next iteration/higher-order
        grad.requires_grad_(True)
        
        current_y = grad  # Chain to next

    return current_y

# ------------------------------
# Unified PDE/ODE Residual
# ------------------------------
def pde_residual(model, inputs, pde_type, **kwargs):
    """
    Compute residual for various ODEs/PDEs automatically.
    """
    
    # FIXED: Clone WITHOUT detach to preserve graph
    inputs_for_model = inputs.clone().requires_grad_(True)
    u_val_raw = model(inputs_for_model)

    # Schrödinger: Special real/imag handling for complex stability
    if pde_type == "schrodinger":
        # Split to raw real/imag [N,2] if needed
        if hasattr(u_val_raw, 'is_complex') and u_val_raw.is_complex():
            u_real = u_val_raw.real.unsqueeze(1)
            u_imag = u_val_raw.imag.unsqueeze(1)
            u_val_raw_split = torch.cat([u_real, u_imag], dim=1)
        elif len(u_val_raw.shape) > 1 and u_val_raw.shape[-1] == 2:
            u_val_raw_split = u_val_raw
        else:
            raise ValueError("Schrödinger requires model output as complex [N] or raw [N,2].")
        
        u_r = u_val_raw_split[..., 0]
        u_i = u_val_raw_split[..., 1]
        u_val = torch.complex(u_r, u_i)
        
        x = inputs_for_model[:,0:1].requires_grad_(True)  # FIXED: Explicit grad on slices
        t = inputs_for_model[:,1:2].requires_grad_(True)
        hbar = kwargs.get("hbar", 1.0); m = kwargs.get("m", 1.0)
        V = kwargs.get("V", None)
        if V is not None:
            V_val = V(inputs_for_model)
            if u_val.is_complex() and not V_val.is_complex():
                V_val = V_val.to(torch.complex64)
        else:
            V_val = torch.zeros_like(u_val, dtype=torch.complex64)
        
        # Derivatives on real/imag separately (stable)
        du_r_dt = derivative(u_r, t)
        du_i_dt = derivative(u_i, t)
        du_dt = torch.complex(du_r_dt, du_i_dt)
        
        d2u_r_dx2 = derivative(u_r, x, 2)
        d2u_i_dx2 = derivative(u_i, x, 2)
        d2u_dx2 = torch.complex(d2u_r_dx2, d2u_i_dx2)
        
        return 1j * hbar * du_dt + (hbar**2 / (2 * m)) * d2u_dx2 - V_val * u_val
    
    # Non-Schrodinger: Standard handling
    if hasattr(u_val_raw, 'is_complex') and u_val_raw.is_complex():
        u_val = u_val_raw
    elif len(u_val_raw.shape) > 1 and u_val_raw.shape[-1] == 2:
        # FIXED: Remove dtype=torch.complex64 (infer automatically for compatibility)
        u_val = torch.complex(u_val_raw[..., 0], u_val_raw[..., 1])
    else:
        u_val = u_val_raw
    
    # ------------------ 1D ODEs ------------------
    if pde_type == "logistic":
        var = inputs_for_model[:,0:1].requires_grad_(True)
        r = kwargs.get("r", 1.0); K = kwargs.get("K", 1.0)
        return derivative(u_val, var) - r * u_val * (1 - u_val / K)
    
    if pde_type == "sho":
        var = inputs_for_model[:,0:1].requires_grad_(True)
        return derivative(u_val, var, 2) + u_val
    
    if pde_type == "damped_ho":
        var = inputs_for_model[:,0:1].requires_grad_(True)
        gamma = kwargs.get("gamma", 0.1)
        d2u = derivative(u_val, var, 2)
        du = derivative(u_val, var)
        return d2u + gamma * du + u_val

    if pde_type == "newton_cooling":
        var = inputs_for_model[:,0:1].requires_grad_(True)
        T_env = kwargs.get("T_env", 25.0); k = kwargs.get("k", 0.1)
        return derivative(u_val, var) + k * (u_val - T_env)
    
    # ------------------ 1D PDEs ------------------
    if pde_type in ["heat", "wave", "burgers", "kdv", "convection_diffusion", "markov"]:
        x = inputs_for_model[:,0:1].requires_grad_(True)  # FIXED: Explicit on slices
        t = inputs_for_model[:,1:2].requires_grad_(True)
        
        # FIXED: Safety for unexpected complex u_val (split like Schrödinger)
        if hasattr(u_val, 'is_complex') and u_val.is_complex():
            u_r = u_val.real
            u_i = u_val.imag
            # Compute derivatives on real/imag, then reconstruct (for stability)
            du_r_dt = derivative(u_r, t)
            du_i_dt = derivative(u_i, t)
            du_dt = torch.complex(du_r_dt, du_i_dt)
            
            du_r_x = derivative(u_r, x)
            du_i_x = derivative(u_i, x)
            du_x = torch.complex(du_r_x, du_i_x)
            
            d2u_r_x2 = derivative(u_r, x, 2)
            d2u_i_x2 = derivative(u_i, x, 2)
            d2u_x2 = torch.complex(d2u_r_x2, d2u_i_x2)
            
            d3u_r_x3 = derivative(u_r, x, 3)
            d3u_i_x3 = derivative(u_i, x, 3)
            d3u_x3 = torch.complex(d3u_r_x3, d3u_i_x3)
        else:
            du_dt = derivative(u_val, t)
            du_x = derivative(u_val, x)
            d2u_x2 = derivative(u_val, x, 2)
            d3u_x3 = derivative(u_val, x, 3)
        
        if pde_type == "heat":
            if 'du_dt' in locals() and du_dt.is_complex():
                return du_dt.real - d2u_x2.real  # Use real parts for real PDEs
            return du_dt - d2u_x2
        
        if pde_type == "wave":
            c = kwargs.get("c", 1.0)
            d2u_dt2 = derivative(u_val, t, 2) if not u_val.is_complex() else torch.complex(derivative(u_r, t, 2), derivative(u_i, t, 2))
            if u_val.is_complex():
                return d2u_dt2.real - c**2 * d2u_x2.real
            return d2u_dt2 - c**2 * d2u_x2
        
        if pde_type == "burgers":
            nu = kwargs.get("nu", 0.01)
            if u_val.is_complex():
                return du_dt.real + u_r * du_x.real - nu * d2u_x2.real  # Use real for Burgers (typically real PDE)
            return du_dt + u_val * du_x - nu * d2u_x2
        
        if pde_type == "kdv":
            alpha = kwargs.get("alpha", 6.0); beta = kwargs.get("beta", 1.0)
            if u_val.is_complex():
                return du_dt.real + alpha * u_r * du_x.real + beta * d3u_x3.real
            return du_dt + alpha * u_val * du_x + beta * d3u_x3
        
        if pde_type == "convection_diffusion":
            c = kwargs.get("c", 1.0); D = kwargs.get("D", 0.01)
            if u_val.is_complex():
                return du_dt.real + c * du_x.real - D * d2u_x2.real
            return du_dt + c * du_x - D * d2u_x2
        
        if pde_type == "markov":
            D = kwargs.get("D", 0.1)
            if u_val.is_complex():
                return du_dt.real - D * d2u_x2.real  # Markov typically real diffusion
            return du_dt - D * d2u_x2

    if pde_type == "planck":
        freq = inputs_for_model[:,0:1].requires_grad_(True)
        T = inputs_for_model[:,1:2].requires_grad_(True)
        exact = kwargs.get("exact_planck")
        if exact is None:
            raise ValueError("Provide exact_planck(freq, T) function for Planck residual")
        return u_val - exact(freq, T)

    if pde_type == "photoelectric":
        freq = inputs_for_model[:,0:1].requires_grad_(True)
        work_func = kwargs.get("work_func", 1.0)
        return u_val - torch.maximum(torch.zeros_like(freq), freq - work_func)
    
    # ------------------ 2D/3D PDEs (similar explicit grad on slices) ------------------
    if pde_type == "laplace":
        x = inputs_for_model[:,0:1].requires_grad_(True)
        y = inputs_for_model[:,1:2].requires_grad_(True)
        return derivative(u_val, x, 2) + derivative(u_val, y, 2)
    
    if pde_type == "poisson":
        x = inputs_for_model[:,0:1].requires_grad_(True)
        y = inputs_for_model[:,1:2].requires_grad_(True)
        f = kwargs.get("f", lambda x,y: torch.zeros_like(x))
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) - f(x, y)
    
    if pde_type == "navier_stokes_2d":
        x = inputs_for_model[:,0:1].requires_grad_(True)
        y = inputs_for_model[:,1:2].requires_grad_(True)
        t = inputs_for_model[:,2:3].requires_grad_(True)
        model_u = kwargs.get("model_u")
        model_v = kwargs.get("model_v")
        model_p = kwargs.get("model_p")
        nu = kwargs.get("nu", 0.01)
        U = model_u(inputs_for_model); V = model_v(inputs_for_model); P = model_p(inputs_for_model)
        res_u = (derivative(U, t) + U*derivative(U, x) + V*derivative(U, y) + derivative(P, x) - 
                 nu*(derivative(U, x,2) + derivative(U, y,2)))
        res_v = (derivative(V, t) + U*derivative(V, x) + V*derivative(V, y) + derivative(P, y) - 
                 nu*(derivative(V, x,2) + derivative(V, y,2)))
        res_cont = derivative(U, x) + derivative(V, y)
        return res_u, res_v, res_cont
    
    if pde_type == "laplace_3d":
        x = inputs_for_model[:,0:1].requires_grad_(True)
        y = inputs_for_model[:,1:2].requires_grad_(True)
        z = inputs_for_model[:,2:3].requires_grad_(True)
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) + derivative(u_val, z, 2)
    
    if pde_type == "poisson_3d":
        x = inputs_for_model[:,0:1].requires_grad_(True)
        y = inputs_for_model[:,1:2].requires_grad_(True)
        z = inputs_for_model[:,2:3].requires_grad_(True)
        f = kwargs.get("f", lambda x,y,z: torch.zeros_like(x))
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) + derivative(u_val, z, 2) - f(x,y,z)
    
    raise ValueError(f"PDE/ODE type '{pde_type}' not implemented")
