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
        # MODIFIED: Handle model input for cases like Planck's Law where model expects more features
        model_input = x
        # Heuristic: If model expects 2 input features but only 1 is given (like Planck example)
        # This assumes the first layer of the model is a Linear layer.
        if isinstance(model, torch.nn.Module) and hasattr(model, 'model') and \
           isinstance(model.model[0], torch.nn.Linear) and \
           model.model[0].in_features == 2 and x.shape[1] == 1:
            # This is specifically for the Planck example where model expects (freq, T_val)
            # and x is just freq. We need to add the T_val back.
            # Using a hardcoded T_val=2.7 from example_planck.py
            T_val_for_plot = torch.full_like(x, 2.7).to(x.device)
            model_input = torch.cat([x, T_val_for_plot], dim=1)
        
        y_pred_raw = model(model_input) # MODIFIED: Use the potentially modified model_input

        if y_pred_raw.is_complex():
            y_pred = y_pred_raw.abs() # Keep as tensor for now
        elif len(y_pred_raw.shape) > 1 and y_pred_raw.shape[-1] == 2: # If model outputs real and imag parts
            y_pred = torch.complex(y_pred_raw[..., 0], y_pred_raw[..., 1]).abs() # Keep as tensor
        else:
            y_pred = y_pred_raw # Keep as tensor

        # MODIFIED: Ensure y_pred is 2D [N, 1] for consistent plotting
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
        
        y_pred = y_pred.cpu().numpy()
    
    plt.figure(figsize=(8,5))
    plt.plot(x.cpu().numpy(), y_pred, label="Predicted", lw=2)
    if exact is not None:
        exact_plot = exact.abs().cpu().numpy() if exact.is_complex() else exact.cpu().numpy()
        # MODIFIED: Ensure exact_plot is 2D [N, 1] for consistent plotting
        if exact_plot.ndim == 1:
            exact_plot = exact_plot[:, None] # Add a new axis
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
# ------------------------------
# 2D / 3D Animation over Time
# ------------------------------
def animate_2d(model, x, t, title="2D PDE Evolution", interval=100, device=None):
    """
    Animate a 2D PDE solution over time
    x: tensor [N] or [N,1] spatial points
    t: tensor [M] or [M,1] time points
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
        # FIXED: Robust reshape – ensure x is [N,1], t_i is [N,1]
        x_plot = x.clone()
        if x_plot.dim() == 1:
            x_plot = x_plot.unsqueeze(1)  # [N] -> [N,1]
        
        t_frame = t[frame]
        if t_frame.dim() == 0:  # Scalar
            t_frame = t_frame.unsqueeze(0).unsqueeze(0)  # Scalar -> [1,1]
        elif t_frame.dim() == 1 and t_frame.shape[0] == 1:
            t_frame = t_frame.unsqueeze(0)  # [1] -> [1,1]
        else:
            t_frame = t_frame.unsqueeze(1)  # [1] or [M,1] slice -> [1,1]
        
        t_i = t_frame.repeat(x_plot.shape[0], 1)  # [1,1] -> [N,1]
        
        xt = torch.cat([x_plot, t_i], dim=1)  # Now safe: [N,1] + [N,1] -> [N,2]
        
        with torch.no_grad():
            y_raw = model(xt)
            if hasattr(y_raw, 'is_complex') and y_raw.is_complex():
                y = y_raw.abs().cpu().numpy()  # Magnitude for complex (e.g., Schrödinger |ψ|)
            elif len(y_raw.shape) > 1 and y_raw.shape[-1] == 2:  # Raw real/imag
                y = torch.complex(y_raw[..., 0], y_raw[..., 1]).abs().cpu().numpy()
            else:
                y = y_raw.cpu().numpy()  # Real-valued (e.g., Markov u)
        
        # FIXED: Safe set_data – squeeze x_plot for 1D plot, ravel y
        line.set_data(x_plot.squeeze().cpu().numpy(), y.ravel())
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
