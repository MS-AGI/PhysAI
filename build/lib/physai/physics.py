import torch

def derivative(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return y

# 1D ODEs
def dy_dx_equals_y(x, model):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - y

def dy_dx_equals_func(x, model, func):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - func(x)

def second_order_ode(x, model):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x, 2) + y

def damped_harmonic_oscillator(x, model, gamma=0.1):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x, 2) + gamma*derivative(y, x) + y

def logistic_growth(x, model, r=1.0, K=1.0):
    x.requires_grad_(True)
    y = model(x)
    return derivative(y, x) - r*y*(1 - y/K)

# 1D PDEs
def heat_equation(u, x, t, model):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) - derivative(u_val, x, 2)

def wave_equation(u, x, t, model, c=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t, 2) - c**2 * derivative(u_val, x, 2)

def burgers_equation(u, x, t, model, nu=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + u_val*derivative(u_val, x) - nu*derivative(u_val, x, 2)

def kdv_equation(u, x, t, model, alpha=6.0, beta=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + alpha*u_val*derivative(u_val, x) + beta*derivative(u_val, x, 3)

def convection_diffusion(u, x, t, model, c=1.0, D=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_val = model(torch.cat([x, t], dim=1))
    return derivative(u_val, t) + c*derivative(u_val, x) - D*derivative(u_val, x, 2)

# 2D PDEs
def laplace_equation(u, x, y, model):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u_val = model(torch.cat([x, y], dim=1))
    return derivative(u_val, x, 2) + derivative(u_val, y, 2)

def poisson_equation(u, x, y, f, model):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u_val = model(torch.cat([x, y], dim=1))
    return derivative(u_val, x, 2) + derivative(u_val, y, 2) - f(x, y)
