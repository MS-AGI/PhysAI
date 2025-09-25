import torch

# ------------------------------
# Utility: derivatives
# ------------------------------
def derivative(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return y

# ------------------------------
# Unified PDE/ODE Residual
# ------------------------------
def pde_residual(model, inputs, pde_type, **kwargs):
    """
    Compute residual for various ODEs/PDEs automatically.
    
    model: torch.nn model
    inputs: tensor of shape [N, D] where D = 1,2,3 (x, y, z / x, t / x, y, t)
    pde_type: str, e.g. "heat", "wave", "burgers", "kdv", "laplace", "poisson", "schrodinger",
                      "navier_stokes_2d", "logistic", "sho", "damped_ho",
                      "markov", "planck", "newton_cooling", "photoelectric"
    kwargs: extra parameters (nu, c, alpha, beta, V, f, r, K, gamma, etc.)
    """
    
    # Enable gradients
    inputs = inputs.clone().detach().requires_grad_(True)
    u_val = model(inputs)
    
    # ------------------ 1D ODEs ------------------
    if pde_type == "logistic":
        r = kwargs.get("r", 1.0)
        K = kwargs.get("K", 1.0)
        return derivative(u_val, inputs[:,0:1]) - r*u_val*(1 - u_val/K)
    
    if pde_type == "sho":
        return derivative(u_val, inputs[:,0:1], 2) + u_val
    
    if pde_type == "damped_ho":
        gamma = kwargs.get("gamma", 0.1)
        return derivative(u_val, inputs[:,0:1], 2) + gamma*derivative(u_val, inputs[:,0:1]) + u_val

    if pde_type == "newton_cooling":
        T_env = kwargs.get("T_env", 25.0)
        k = kwargs.get("k", 0.1)
        return derivative(u_val, inputs[:,0:1]) + k*(u_val - T_env)
    
    # ------------------ 1D PDEs ------------------
    if pde_type == "heat":
        x = inputs[:,0:1]; t = inputs[:,1:2]
        return derivative(u_val, t) - derivative(u_val, x, 2)
    
    if pde_type == "wave":
        x = inputs[:,0:1]; t = inputs[:,1:2]
        c = kwargs.get("c", 1.0)
        return derivative(u_val, t, 2) - c**2 * derivative(u_val, x, 2)
    
    if pde_type == "burgers":
        x = inputs[:,0:1]; t = inputs[:,1:2]; nu = kwargs.get("nu", 0.01)
        return derivative(u_val, t) + u_val*derivative(u_val, x) - nu*derivative(u_val, x, 2)
    
    if pde_type == "kdv":
        x = inputs[:,0:1]; t = inputs[:,1:2]
        alpha = kwargs.get("alpha", 6.0)
        beta  = kwargs.get("beta", 1.0)
        return derivative(u_val, t) + alpha*u_val*derivative(u_val, x) + beta*derivative(u_val, x, 3)
    
    if pde_type == "convection_diffusion":
        x = inputs[:,0:1]; t = inputs[:,1:2]
        c = kwargs.get("c", 1.0); D = kwargs.get("D", 0.01)
        return derivative(u_val, t) + c*derivative(u_val, x) - D*derivative(u_val, x, 2)
    
    if pde_type == "markov":
        x = inputs[:,0:1]; t = inputs[:,1:2]
        D = kwargs.get("D", 0.1)
        return derivative(u_val, t) - D*derivative(u_val, x, 2)

    if pde_type == "planck":
        freq = inputs[:,0:1]; T = inputs[:,1:2]
        exact = kwargs.get("exact_planck")
        if exact is None:
            raise ValueError("Provide exact_planck(freq, T) function for Planck residual")
        return u_val - exact(freq, T)

    if pde_type == "photoelectric":
        freq = inputs[:,0:1]; work_func = kwargs.get("work_func", 1.0)
        return u_val - torch.maximum(torch.zeros_like(freq), freq - work_func)
    
    # ------------------ 2D PDEs ------------------
    if pde_type == "laplace":
        x, y = inputs[:,0:1], inputs[:,1:2]
        return derivative(u_val, x, 2) + derivative(u_val, y, 2)
    
    if pde_type == "poisson":
        x, y = inputs[:,0:1], inputs[:,1:2]
        f = kwargs.get("f", lambda x,y: torch.zeros_like(x))
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) - f(x, y)
    
    if pde_type == "schrodinger":
        x, t = inputs[:,0:1], inputs[:,1:2]
        hbar = kwargs.get("hbar", 1.0); m = kwargs.get("m", 1.0)
        V = kwargs.get("V", None)
        V_val = V(inputs) if V is not None else torch.zeros_like(u_val)
        return 1j*hbar*derivative(u_val, t) + (hbar**2/(2*m))*derivative(u_val, x, 2) - V_val*u_val
    
    if pde_type == "navier_stokes_2d":
        x, y, t = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        model_u = kwargs.get("model_u")
        model_v = kwargs.get("model_v")
        model_p = kwargs.get("model_p")
        nu = kwargs.get("nu", 0.01)
        U = model_u(inputs); V = model_v(inputs); P = model_p(inputs)
        res_u = derivative(U, t) + U*derivative(U, x) + V*derivative(U, y) + derivative(P, x) - nu*(derivative(U, x,2)+derivative(U, y,2))
        res_v = derivative(V, t) + U*derivative(V, x) + V*derivative(V, y) + derivative(P, y) - nu*(derivative(V, x,2)+derivative(V, y,2))
        res_cont = derivative(U, x) + derivative(V, y)
        return res_u, res_v, res_cont
    
    # ------------------ 3D PDEs ------------------
    if pde_type == "laplace_3d":
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) + derivative(u_val, z, 2)
    
    if pde_type == "poisson_3d":
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        f = kwargs.get("f", lambda x,y,z: torch.zeros_like(x))
        return derivative(u_val, x, 2) + derivative(u_val, y, 2) + derivative(u_val, z, 2) - f(x,y,z)
    
    raise ValueError(f"PDE/ODE type '{pde_type}' not implemented")
