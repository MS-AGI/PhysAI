import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------
# 1D plotting for ODEs / PDEs
# ------------------------------
def plot_1d_solution(model, x, exact=None, title="1D Solution", device=None):
    """
    Plot 1D solution of a model or function.
    model: torch model or callable
    x: tensor of shape [N,1]
    exact: optional exact solution for comparison
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    
    with torch.no_grad():
        y_pred_raw = model(x)
        if y_pred_raw.is_complex():
            y_pred = y_pred_raw.abs().cpu().numpy() # Plot magnitude for complex solutions
        elif y_pred_raw.shape[-1] == 2: # If model outputs real and imag parts
            y_pred = torch.complex(y_pred_raw[..., 0], y_pred_raw[..., 1]).abs().cpu().numpy()
        else:
            y_pred = y_pred_raw.cpu().numpy()
    
    plt.figure(figsize=(8,5))
    plt.plot(x.cpu().numpy(), y_pred, label="Predicted", lw=2)
    if exact is not None:
        exact_plot = exact.abs().cpu().numpy() if exact.is_complex() else exact.cpu().numpy()
        plt.plot(x.cpu().numpy(), exact_plot, "--", label="Exact", lw=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# 2D Surface Plotting
# ------------------------------
def plot_2d_surface(model, X, Y, title="2D Surface", device=None):
    """
    Plot 2D surface of a model: inputs X,Y tensors of shape [N,1]
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    Y = Y.to(device)
    XY = torch.cat([X, Y], dim=1)
    
    with torch.no_grad():
        Z_raw = model(XY)
        if Z_raw.is_complex():
            Z = Z_raw.abs().cpu().numpy() # Plot magnitude for complex solutions
        elif Z_raw.shape[-1] == 2: # If model outputs real and imag parts
            Z = torch.complex(Z_raw[..., 0], Z_raw[..., 1]).abs().cpu().numpy()
        else:
            Z = Z_raw.cpu().numpy()
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X.cpu().numpy().ravel(), Y.cpu().numpy().ravel(), Z.ravel(), cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("U")
    ax.set_title(title)
    plt.show()


# ------------------------------
# 2D / 3D Animation over Time
# ------------------------------
def animate_2d(model, x, t, title="2D PDE Evolution", interval=100, device=None):
    """
    Animate a 2D PDE solution over time
    x: tensor [N,1] spatial points
    t: tensor [M,1] time points
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    t = t.to(device)
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    
    ax.set_xlim(float(x.min()), float(x.max()))
    # Adjust ylim dynamically or based on expected range
    # For Schrödinger, plotting magnitude, so range is [0, max_magnitude]
    ax.set_ylim(0, 1.0) # Default to [0, 1] for magnitude, adjust as needed
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(title)
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(frame):
        t_i = t[frame].repeat(x.shape[0],1)
        xt = torch.cat([x, t_i], dim=1)
        with torch.no_grad():
            y_raw = model(xt)
            if y_raw.is_complex():
                y = y_raw.abs().cpu().numpy()
            elif y_raw.shape[-1] == 2: # If model outputs real and imag parts
                y = torch.complex(y_raw[..., 0], y_raw[..., 1]).abs().cpu().numpy()
            else:
                y = y_raw.cpu().numpy()
        line.set_data(x.cpu().numpy(), y.ravel())
        ax.set_title(f"{title} | t={t[frame].item():.2f}")
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval)
    plt.show()
    return ani


# ------------------------------
# Loss Plotting
# ------------------------------
def plot_loss(history, title="Training Loss"):
    """
    Plot training loss curves
    history: dict with keys 'total_loss', 'res_loss', 'bc_loss'
    """
    plt.figure(figsize=(8,5))
    plt.plot(history["total_loss"], label="Total Loss", lw=2)
    plt.plot(history["res_loss"], label="Residual Loss", lw=2)
    plt.plot(history["bc_loss"], label="BC Loss", lw=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# Optional callback for Trainer
# ------------------------------
def visualization_callback(model, x, t=None, kind="1d", interval=100, device=None):
    """
    General callback function to visualize PINN during training.
    kind: "1d", "2d_surface", "2d_animation"
    x: spatial points
    t: optional time points
    """
    if kind == "1d":
        plot_1d_solution(model, x, title=f"{model.__class__.__name__} 1D Solution", device=device)
    elif kind == "2d_surface":
        assert t is not None, "Provide Y grid for 2D surface"
        plot_2d_surface(model, x, t, title=f"{model.__class__.__name__} 2D Surface", device=device)
    elif kind == "2d_animation":
        assert t is not None, "Provide time points for animation"
        animate_2d(model, x, t, title=f"{model.__class__.__name__} 2D Animation", interval=interval, device=device)
