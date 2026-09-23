"""
physai/visualization.py

Publication-quality visualisation utilities for PhysAI training results
and PDE solutions.

All plot functions return a ``(fig, axes)`` tuple so callers can further
customise or save.  They never call ``plt.show()`` — the caller controls
display.

Functions
---------
plot_loss_history          — multi-term loss curves on log scale
plot_solution_1d           — 1-D PDE solution vs. reference
plot_solution_2d           — 2-D heatmap / contour of PDE solution
plot_solution_2d_comparison — side-by-side: prediction | reference | error
plot_residual_field        — spatial distribution of PDE residual magnitude
plot_collocation_points    — scatter of collocation / BC / IC points
plot_spectrum              — energy spectrum of predicted field (FFT-based)
plot_training_animation    — saves a GIF of the solution evolving over steps
animate_1d_solution        — animates a 1-D solution over a parameter sweep
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Default style
# ---------------------------------------------------------------------------

_PHYSAI_STYLE: Dict[str, Any] = {
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "axes.labelcolor":      "#111111",
    "axes.titlesize":       13,
    "axes.labelsize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "grid.color":           "#DDDDDD",
    "grid.linewidth":       0.6,
    "lines.linewidth":      2.0,
    "font.family":          "DejaVu Sans",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.frameon":       False,
    "legend.fontsize":      9,
}


def _apply_style() -> None:
    _require_mpl()
    plt.rcParams.update(_PHYSAI_STYLE)


# ---------------------------------------------------------------------------
# 1. Loss history
# ---------------------------------------------------------------------------

def plot_loss_history(
    history: Any,
    *,
    terms: Optional[List[str]] = None,
    log_scale: bool = True,
    figsize: Tuple[float, float] = (9, 4),
    title: str = "Training Loss History",
    smoothing: int = 1,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot total loss and per-term loss curves from a TrainingHistory.

    Parameters
    ----------
    history   : TrainingHistory (or any object with .steps / .total_loss / .terms)
    terms     : subset of term names to plot; defaults to all
    log_scale : use log-y axis
    smoothing : moving-average window (1 = no smoothing)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    steps = np.array(history.steps)

    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    def _trim_steps(s: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return s
        pad = w - 1
        return s[pad // 2: len(s) - (pad - pad // 2)]

    total = np.array(history.total_loss)
    s_steps = _trim_steps(steps, smoothing)

    ax.plot(
        s_steps,
        _smooth(total, smoothing),
        label="total",
        color="#1f77b4",
        linewidth=2.2,
        zorder=5,
    )

    # Per-term curves
    _colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    term_names = terms or list(history.terms.keys())
    for i, name in enumerate(term_names):
        if name not in history.terms:
            warnings.warn(f"Term '{name}' not found in history.")
            continue
        vals = np.array(history.terms[name])
        ax.plot(
            s_steps,
            _smooth(vals, smoothing),
            label=name,
            color=_colors[(i + 1) % len(_colors)],
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
        )

    # Validation loss (if present)
    if history.val_loss:
        val_steps = steps[:: len(steps) // max(len(history.val_loss), 1)][:len(history.val_loss)]
        ax.plot(
            val_steps,
            np.array(history.val_loss),
            label="validation",
            color="#d62728",
            linewidth=1.8,
            linestyle=":",
            marker="o",
            markersize=3,
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(ncol=min(len(term_names) + 2, 4))
    ax.grid(True, which="both", alpha=0.5)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. 1-D solution
# ---------------------------------------------------------------------------

def plot_solution_1d(
    x: np.ndarray,
    prediction: np.ndarray,
    reference: Optional[np.ndarray] = None,
    *,
    xlabel: str = "x",
    ylabel: str = "u(x)",
    title: str = "1-D PDE Solution",
    figsize: Tuple[float, float] = (8, 4),
    show_error: bool = True,
) -> Tuple["plt.Figure", Union["plt.Axes", np.ndarray]]:
    """
    Plot a 1-D PDE solution with optional reference and pointwise error.
    """
    _apply_style()
    n_rows = 2 if (reference is not None and show_error) else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize[0], figsize[1] * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    ax = axes[0]
    ax.plot(x, prediction, label="PhysAI prediction", color="#1f77b4", linewidth=2.0)
    if reference is not None:
        ax.plot(
            x, reference,
            label="Reference",
            color="#ff7f0e",
            linewidth=1.6,
            linestyle="--",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)

    if reference is not None and show_error and n_rows == 2:
        err = np.abs(prediction - reference)
        ax2 = axes[1]
        ax2.semilogy(x, err + 1e-16, color="#d62728", linewidth=1.6)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("|Error|")
        ax2.set_title("Pointwise absolute error")
        ax2.grid(True, which="both", alpha=0.4)
    else:
        axes[-1].set_xlabel(xlabel)

    fig.tight_layout()
    return fig, axes if len(axes) > 1 else axes[0]


# ---------------------------------------------------------------------------
# 3. 2-D solution heatmap
# ---------------------------------------------------------------------------

def plot_solution_2d(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    clabel: str = "u",
    title: str = "2-D PDE Solution",
    figsize: Tuple[float, float] = (7, 5),
    cmap: str = "RdBu_r",
    symmetric_cbar: bool = False,
    n_contours: int = 20,
    show_contours: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Heatmap + optional contour overlay of a 2-D scalar field.

    Parameters
    ----------
    x, y  : 1-D coordinate arrays (meshgrid components accepted too)
    field : 2-D array of shape (len(y), len(x))
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    if symmetric_cbar:
        vmax = float(np.max(np.abs(field)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=float(field.min()), vmax=float(field.max()))

    im = ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading="auto")
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax, label=clabel)

    if show_contours:
        ax.contour(x, y, field, levels=n_contours, colors="k", linewidths=0.4, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Side-by-side comparison
# ---------------------------------------------------------------------------

def plot_solution_2d_comparison(
    x: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Prediction vs. Reference",
    figsize: Tuple[float, float] = (15, 4),
    cmap: str = "RdBu_r",
    error_log_scale: bool = False,
) -> Tuple["plt.Figure", np.ndarray]:
    """
    Three-panel figure: Prediction | Reference | Absolute Error.
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    vmax = max(float(np.abs(prediction).max()), float(np.abs(reference).max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    err  = np.abs(prediction - reference)

    panels = [
        ("Prediction",      prediction, norm,                           cmap),
        ("Reference",       reference,  norm,                           cmap),
        ("Absolute Error",  err,
         LogNorm(vmin=max(err.min(), 1e-16), vmax=err.max())
         if error_log_scale
         else Normalize(vmin=0, vmax=err.max()),
         "Reds"),
    ]

    axes = []
    for col, (ttl, data, n, cm) in enumerate(panels):
        ax  = fig.add_subplot(gs[0, col])
        im  = ax.pcolormesh(x, y, data, cmap=cm, norm=n, shading="auto")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.06)
        fig.colorbar(im, cax=cax)
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        axes.append(ax)

    fig.suptitle(title, fontsize=14, y=1.01)
    return fig, np.array(axes)


# ---------------------------------------------------------------------------
# 5. Residual field
# ---------------------------------------------------------------------------

def plot_residual_field(
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "PDE Residual Magnitude",
    figsize: Tuple[float, float] = (7, 5),
    log_scale: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Heatmap of |residual| over the domain; useful for diagnosing where
    the model struggles.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    mag  = np.abs(residual)
    if log_scale:
        norm = LogNorm(vmin=max(mag.min(), 1e-16), vmax=mag.max())
    else:
        norm = Normalize(vmin=0, vmax=mag.max())

    im  = ax.pcolormesh(x, y, mag, cmap="hot_r", norm=norm, shading="auto")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax, label="|residual|")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Collocation scatter
# ---------------------------------------------------------------------------

def plot_collocation_points(
    collocation: np.ndarray,
    bc_points: Optional[np.ndarray] = None,
    ic_points: Optional[np.ndarray] = None,
    data_points: Optional[np.ndarray] = None,
    *,
    dim_x: int = 0,
    dim_y: int = 1,
    xlabel: str = "x",
    ylabel: str = "y / t",
    title: str = "Collocation & Boundary Points",
    figsize: Tuple[float, float] = (7, 5),
    alpha: float = 0.25,
    point_size: float = 6.0,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Scatter plot of all point sets used during training.
    Supports arbitrary-dimensional inputs by projecting onto (dim_x, dim_y).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    def _scatter(pts: np.ndarray, label: str, color: str, zorder: int) -> None:
        ax.scatter(
            pts[:, dim_x], pts[:, dim_y],
            s      = point_size,
            color  = color,
            label  = label,
            alpha  = alpha,
            zorder = zorder,
            linewidths = 0,
        )

    _scatter(collocation, f"Collocation ({len(collocation)})", "#1f77b4", 1)
    if bc_points is not None:
        _scatter(bc_points, f"BC ({len(bc_points)})", "#d62728", 3)
    if ic_points is not None:
        _scatter(ic_points, f"IC ({len(ic_points)})", "#2ca02c", 2)
    if data_points is not None:
        _scatter(data_points, f"Data ({len(data_points)})", "#9467bd", 4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    field: np.ndarray,
    dx: float = 1.0,
    *,
    title: str = "Energy Spectrum",
    figsize: Tuple[float, float] = (7, 4),
    label: str = "predicted",
    reference_field: Optional[np.ndarray] = None,
    reference_label: str = "reference",
    kolmogorov_slope: bool = False,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot the radially-averaged energy spectrum E(k) = ½|û(k)|² of a 2-D field.
    For 1-D fields, the power spectrum |û(k)|² is plotted directly.

    Parameters
    ----------
    field     : 1-D or 2-D numpy array
    dx        : grid spacing (for proper wavenumber axis)
    kolmogorov_slope : overlay -5/3 reference slope
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    def _spectrum_1d(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N    = len(f)
        fhat = np.fft.rfft(f)
        k    = np.fft.rfftfreq(N, d=dx) * N
        E    = 0.5 * np.abs(fhat) ** 2 / N ** 2
        return k[1:], E[1:]

    def _spectrum_2d(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ny, Nx = f.shape
        fhat   = np.fft.fft2(f)
        kx     = np.fft.fftfreq(Nx, d=dx) * Nx
        ky     = np.fft.fftfreq(Ny, d=dx) * Ny
        KX, KY = np.meshgrid(kx, ky)
        K      = np.sqrt(KX ** 2 + KY ** 2)
        E2     = 0.5 * np.abs(fhat) ** 2 / (Nx * Ny) ** 2
        k_bins = np.arange(1, min(Nx, Ny) // 2)
        E_bins = np.array([
            E2[(K >= k - 0.5) & (K < k + 0.5)].sum()
            for k in k_bins
        ])
        return k_bins.astype(float), E_bins

    if field.ndim == 1:
        k, E = _spectrum_1d(field)
    else:
        k, E = _spectrum_2d(field)

    ax.loglog(k, E, label=label, color="#1f77b4", linewidth=2.0)

    if reference_field is not None:
        if reference_field.ndim == 1:
            kr, Er = _spectrum_1d(reference_field)
        else:
            kr, Er = _spectrum_2d(reference_field)
        ax.loglog(kr, Er, label=reference_label, color="#ff7f0e",
                  linewidth=1.6, linestyle="--")

    if kolmogorov_slope:
        k_ref = k[len(k) // 4 : 3 * len(k) // 4]
        C     = float(E[len(k) // 4]) * k_ref[0] ** (5 / 3)
        ax.loglog(k_ref, C * k_ref ** (-5 / 3), "k--", linewidth=1.0,
                  alpha=0.6, label=r"$k^{-5/3}$")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 8. Animated 1-D solution
# ---------------------------------------------------------------------------

def animate_1d_solution(
    x: np.ndarray,
    frames: np.ndarray,
    *,
    reference_frames: Optional[np.ndarray] = None,
    param_values: Optional[np.ndarray] = None,
    param_label: str = "t",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    title: str = "PhysAI — Solution evolution",
    figsize: Tuple[float, float] = (8, 4),
    fps: int = 20,
    save_path: Optional[str] = None,
    interval_ms: int = 50,
) -> "FuncAnimation":
    """
    Animate a 1-D solution over a sequence of frames (time steps or
    parameter values).

    Parameters
    ----------
    x               : 1-D coordinate array [N_x]
    frames          : predicted solution array [N_frames, N_x]
    reference_frames: reference solution array [N_frames, N_x]  (optional)
    param_values    : parameter value for each frame (for title display)
    save_path       : if provided, save animation as GIF at this path
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ymin = float(frames.min()) - 0.1 * abs(float(frames.min()))
    ymax = float(frames.max()) + 0.1 * abs(float(frames.max()))
    if reference_frames is not None:
        ymin = min(ymin, float(reference_frames.min()))
        ymax = max(ymax, float(reference_frames.max()))

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)

    (line_pred,) = ax.plot([], [], color="#1f77b4", linewidth=2.0, label="PhysAI")
    line_ref = None
    if reference_frames is not None:
        (line_ref,) = ax.plot([], [], color="#ff7f0e", linewidth=1.6,
                              linestyle="--", label="Reference")
    ax.legend()

    def _init():
        line_pred.set_data([], [])
        if line_ref is not None:
            line_ref.set_data([], [])
        return (line_pred,) if line_ref is None else (line_pred, line_ref)

    def _update(frame_idx: int):
        line_pred.set_data(x, frames[frame_idx])
        if line_ref is not None:
            line_ref.set_data(x, reference_frames[frame_idx])
        pv = (
            f"{param_label} = {param_values[frame_idx]:.4g}"
            if param_values is not None
            else f"frame {frame_idx}"
        )
        ax.set_title(f"{title}   [{pv}]")
        return (line_pred,) if line_ref is None else (line_pred, line_ref)

    anim = FuncAnimation(
        fig,
        _update,
        frames      = len(frames),
        init_func   = _init,
        interval    = interval_ms,
        blit        = True,
    )

    if save_path is not None:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)

    return anim


# ---------------------------------------------------------------------------
# 9. Training progress animation (loss + snapshot)
# ---------------------------------------------------------------------------

def plot_training_animation(
    x: np.ndarray,
    snapshots: List[np.ndarray],
    loss_history: List[float],
    *,
    snapshot_steps: Optional[List[int]] = None,
    xlabel: str = "x",
    ylabel: str = "u(x)",
    figsize: Tuple[float, float] = (12, 4),
    fps: int = 8,
    save_path: Optional[str] = None,
) -> "FuncAnimation":
    """
    Dual-panel animation: loss curve (growing) on the left, current
    solution snapshot on the right.

    Parameters
    ----------
    x              : spatial coordinates
    snapshots      : list of solution arrays, one per snapshot step
    loss_history   : full loss history (same length as snapshots)
    snapshot_steps : step indices corresponding to snapshots
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_sol  = fig.add_subplot(gs[0, 1])

    N = len(snapshots)
    steps = snapshot_steps or list(range(N))
    loss_arr = np.array(loss_history)

    ymin_s = min(s.min() for s in snapshots) * 1.1
    ymax_s = max(s.max() for s in snapshots) * 1.1

    ax_sol.set_xlim(float(x.min()), float(x.max()))
    ax_sol.set_ylim(ymin_s, ymax_s)
    ax_sol.set_xlabel(xlabel)
    ax_sol.set_ylabel(ylabel)
    ax_sol.grid(True, alpha=0.35)

    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, steps[-1])
    ax_loss.set_ylim(loss_arr[loss_arr > 0].min() * 0.5, loss_arr.max() * 2)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, which="both", alpha=0.4)

    (loss_line,) = ax_loss.plot([], [], color="#1f77b4", linewidth=1.8)
    loss_dot,    = ax_loss.plot([], [], "o", color="#d62728", markersize=5)
    (sol_line,)  = ax_sol.plot([], [], color="#1f77b4", linewidth=2.0)

    def _init():
        loss_line.set_data([], [])
        loss_dot.set_data([], [])
        sol_line.set_data([], [])
        return loss_line, loss_dot, sol_line

    def _update(i: int):
        s = steps[i]
        loss_line.set_data(steps[:i + 1], loss_arr[:i + 1])
        loss_dot.set_data([s], [loss_arr[i]])
        sol_line.set_data(x, snapshots[i])
        ax_sol.set_title(f"Solution at step {s}")
        return loss_line, loss_dot, sol_line

    anim = FuncAnimation(
        fig, _update, frames=N, init_func=_init,
        interval=int(1000 / fps), blit=True,
    )

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=fps))

    return anim


# ---------------------------------------------------------------------------
# 10. Error convergence panel
# ---------------------------------------------------------------------------

def plot_error_convergence(
    grid_sizes: Sequence[int],
    errors: Sequence[float],
    *,
    reference_slopes: Optional[Dict[str, float]] = None,
    xlabel: str = "Grid size N",
    ylabel: str = "L² error",
    title: str = "Error convergence",
    figsize: Tuple[float, float] = (6, 4),
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Log-log convergence plot for mesh refinement studies.

    Parameters
    ----------
    reference_slopes : dict of {label: slope}  e.g. {"order-2": -2.0}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    N   = np.array(grid_sizes, dtype=float)
    err = np.array(errors, dtype=float)

    ax.loglog(N, err, "o-", color="#1f77b4", linewidth=2.0, markersize=5,
              label="PhysAI")

    if reference_slopes:
        for label, slope in reference_slopes.items():
            # Anchor to first data point
            C    = err[0] / (N[0] ** slope)
            ref  = C * N ** slope
            ax.loglog(N, ref, "--", linewidth=1.2, alpha=0.75, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "plot_loss_history",
    "plot_solution_1d",
    "plot_solution_2d",
    "plot_solution_2d_comparison",
    "plot_residual_field",
    "plot_collocation_points",
    "plot_spectrum",
    "animate_1d_solution",
    "plot_training_animation",
    "plot_error_convergence",
]