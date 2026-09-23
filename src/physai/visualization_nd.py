"""
physai/visualization_nd.py

N-dimensional visualisation and animation utilities for PhysAI.

The existing ``physai.visualization`` module only understands 1-D and
2-D fields. This module adds general support for fields of arbitrary
dimensionality (3-D volumes, 4-D spatio-temporal fields, 5-D+ parameter
sweeps, etc.) by combining three strategies:

1. **Slicing** — fix all but 1 or 2 axes at a given index and view the
   resulting 1-D/2-D cross-section.
2. **Projection** — reduce extra axes with an aggregation (mean, max,
   sum, rms) to collapse an N-D field down to something plottable.
3. **Animation** — sweep one axis (time, a physical parameter, or a
   spatial slice index) as an animation, optionally holding other axes
   fixed or projected.

All functions accept a plain ``numpy.ndarray`` of shape
``(N0, N1, ..., N_{d-1})`` plus a list of coordinate arrays / labels
for each axis, so they work for any PDE regardless of spatial
dimension. 3-D scalar fields additionally get proper volumetric
rendering (isosurfaces / volume rendering) via ``plotly`` when it is
installed, with a matplotlib voxel/scatter fallback otherwise.

Functions
---------
describe_field            — quick shape/axis summary of an N-D array
slice_field                — extract a 1-D/2-D slice from an N-D field
project_field               — reduce an N-D field along given axes
plot_slice                 — plot a 1-D or 2-D slice of an N-D field
plot_slice_grid             — small-multiples grid of slices along one axis
plot_isosurface_3d          — isosurface / volume rendering of a 3-D field
animate_nd_field            — animate an N-D field by sweeping one axis
animate_isosurface_3d       — animate a 4-D field (3-D volume + time) as
                               a rotating / evolving isosurface
interactive_nd_explorer     — Jupyter widget slice explorer (optional dep)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive 3-D isosurface/volume "
            "rendering. Install with: pip install plotly"
        )


# ---------------------------------------------------------------------------
# Shared style (kept consistent with physai.visualization)
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
# Axis bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class AxisSpec:
    """Describes one axis of an N-D field."""
    name: str
    coords: Optional[np.ndarray] = None  # 1-D coordinate values, len == field.shape[axis]

    def label(self) -> str:
        return self.name


def _make_axis_specs(
    ndim: int,
    shape: Sequence[int],
    axis_names: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> List[AxisSpec]:
    if axis_names is None:
        # sensible defaults: x, y, z, then w0, w1, ...
        base = ["x", "y", "z"]
        axis_names = [
            base[i] if i < len(base) else f"w{i - len(base)}"
            for i in range(ndim)
        ]
    if len(axis_names) != ndim:
        raise ValueError(f"axis_names has length {len(axis_names)}, field has {ndim} dims")

    if coords is None:
        coords = [None] * ndim
    if len(coords) != ndim:
        raise ValueError(f"coords has length {len(coords)}, field has {ndim} dims")

    specs = []
    for i in range(ndim):
        c = coords[i]
        if c is None:
            c = np.arange(shape[i])
        else:
            c = np.asarray(c)
            if c.shape[0] != shape[i]:
                raise ValueError(
                    f"coords[{i}] has length {c.shape[0]}, "
                    f"expected {shape[i]} to match field axis {i}"
                )
        specs.append(AxisSpec(name=axis_names[i], coords=c))
    return specs


def describe_field(
    field: np.ndarray,
    axis_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Human-readable summary of an N-D field's shape, axes, and value range.
    Useful before deciding how to slice/project/animate it.
    """
    specs = _make_axis_specs(field.ndim, field.shape, axis_names)
    lines = [f"Field: shape={field.shape}  dtype={field.dtype}  ndim={field.ndim}"]
    for i, sp in enumerate(specs):
        c = sp.coords
        lines.append(
            f"  axis {i} '{sp.name}': size={field.shape[i]}, "
            f"range=[{float(c[0]):.4g}, {float(c[-1]):.4g}]"
        )
    lines.append(
        f"  value range: [{float(np.nanmin(field)):.4g}, {float(np.nanmax(field)):.4g}]"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Slicing
# ---------------------------------------------------------------------------

def slice_field(
    field: np.ndarray,
    keep_axes: Sequence[int],
    fixed_indices: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """
    Extract a lower-dimensional slice from an N-D field.

    Parameters
    ----------
    field         : N-D array
    keep_axes     : which axes survive in the output (1 or 2 typical)
    fixed_indices : {axis: index} for every axis NOT in keep_axes.
                    Axes not in keep_axes and not given an index default
                    to the middle index of that axis.

    Returns
    -------
    A ``len(keep_axes)``-dimensional array, with axes ordered as given
    in ``keep_axes``.
    """
    ndim = field.ndim
    fixed_indices = dict(fixed_indices or {})
    other_axes = [a for a in range(ndim) if a not in keep_axes]

    indexer: List[Union[int, slice]] = [slice(None)] * ndim
    for a in other_axes:
        idx = fixed_indices.get(a, field.shape[a] // 2)
        indexer[a] = idx

    sliced = field[tuple(indexer)]
    # After integer-indexing, remaining dims are exactly len(keep_axes),
    # in the *original* relative order. Reorder to match `keep_axes`.
    remaining_axes_in_order = [a for a in range(ndim) if a in keep_axes]
    if list(keep_axes) != remaining_axes_in_order:
        perm = [remaining_axes_in_order.index(a) for a in keep_axes]
        sliced = np.transpose(sliced, perm)
    return sliced


# ---------------------------------------------------------------------------
# 2. Projection / reduction
# ---------------------------------------------------------------------------

_REDUCERS: Dict[str, Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]] = {
    "mean": lambda f, ax: np.nanmean(f, axis=ax),
    "max":  lambda f, ax: np.nanmax(f, axis=ax),
    "min":  lambda f, ax: np.nanmin(f, axis=ax),
    "sum":  lambda f, ax: np.nansum(f, axis=ax),
    "rms":  lambda f, ax: np.sqrt(np.nanmean(f ** 2, axis=ax)),
    "std":  lambda f, ax: np.nanstd(f, axis=ax),
}


def project_field(
    field: np.ndarray,
    keep_axes: Sequence[int],
    reduction: str = "mean",
) -> np.ndarray:
    """
    Collapse an N-D field down to ``len(keep_axes)`` dimensions by
    reducing over all other axes.

    Parameters
    ----------
    reduction : one of "mean", "max", "min", "sum", "rms", "std"
    """
    if reduction not in _REDUCERS:
        raise ValueError(f"Unknown reduction '{reduction}'. Choose from {list(_REDUCERS)}")
    ndim = field.ndim
    reduce_axes = tuple(a for a in range(ndim) if a not in keep_axes)
    projected = _REDUCERS[reduction](field, reduce_axes) if reduce_axes else field
    # np reduction over multiple axes preserves the order of remaining axes
    remaining_axes_in_order = [a for a in range(ndim) if a in keep_axes]
    if list(keep_axes) != remaining_axes_in_order:
        perm = [remaining_axes_in_order.index(a) for a in keep_axes]
        projected = np.transpose(projected, perm)
    return projected


# ---------------------------------------------------------------------------
# 3. Static slice plot (1-D line or 2-D heatmap), from an N-D field
# ---------------------------------------------------------------------------

def plot_slice(
    field: np.ndarray,
    keep_axes: Sequence[int],
    *,
    mode: str = "slice",
    reduction: str = "mean",
    fixed_indices: Optional[Dict[int, int]] = None,
    axis_names: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    cmap: str = "RdBu_r",
    symmetric_cbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot a 1-D or 2-D view of an arbitrary N-D field.

    Parameters
    ----------
    field         : N-D array
    keep_axes     : 1 or 2 axis indices to visualise
    mode          : "slice" (fix other axes at an index, default = middle)
                    or "project" (reduce other axes with `reduction`)
    reduction     : reducer used when mode="project"
    fixed_indices : indices for the non-kept axes when mode="slice"
    axis_names    : names for every axis of `field` (len == field.ndim)
    coords        : coordinate arrays for every axis (len == field.ndim)

    Returns
    -------
    (fig, ax)
    """
    _apply_style()
    if len(keep_axes) not in (1, 2):
        raise ValueError("keep_axes must have length 1 or 2")

    specs = _make_axis_specs(field.ndim, field.shape, axis_names, coords)

    if mode == "slice":
        data = slice_field(field, keep_axes, fixed_indices)
        other_axes = [a for a in range(field.ndim) if a not in keep_axes]
        fixed_indices = dict(fixed_indices or {})
        fixed_desc = ", ".join(
            f"{specs[a].name}="
            f"{specs[a].coords[fixed_indices.get(a, field.shape[a] // 2)]:.3g}"
            for a in other_axes
        )
        mode_desc = f"slice @ {fixed_desc}" if fixed_desc else "slice"
    elif mode == "project":
        data = project_field(field, keep_axes, reduction)
        other_axes = [a for a in range(field.ndim) if a not in keep_axes]
        reduced_desc = ", ".join(specs[a].name for a in other_axes)
        mode_desc = f"{reduction} over ({reduced_desc})" if reduced_desc else reduction
    else:
        raise ValueError("mode must be 'slice' or 'project'")

    if len(keep_axes) == 1:
        ax0 = specs[keep_axes[0]]
        figsize = figsize or (8, 4)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ax0.coords, data, color="#1f77b4", linewidth=2.0)
        ax.set_xlabel(ax0.label())
        ax.set_ylabel("field value")
        ax.grid(True, alpha=0.4)
        ax.set_title(title or f"1-D view — {mode_desc}")
        fig.tight_layout()
        return fig, ax

    ax0, ax1 = specs[keep_axes[0]], specs[keep_axes[1]]
    figsize = figsize or (7, 5)
    fig, ax = plt.subplots(figsize=figsize)

    if symmetric_cbar:
        vmax = float(np.nanmax(np.abs(data))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))

    # data has shape (len(ax0.coords), len(ax1.coords)); pcolormesh wants (Y, X)
    im = ax.pcolormesh(ax1.coords, ax0.coords, data, cmap=cmap, norm=norm, shading="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax, label="field value")

    ax.set_xlabel(ax1.label())
    ax.set_ylabel(ax0.label())
    ax.set_title(title or f"2-D view [{ax0.name}, {ax1.name}] — {mode_desc}")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Small-multiples grid of slices along one axis
# ---------------------------------------------------------------------------

def plot_slice_grid(
    field: np.ndarray,
    keep_axes: Sequence[int],
    sweep_axis: int,
    *,
    n_panels: int = 6,
    fixed_indices: Optional[Dict[int, int]] = None,
    axis_names: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    cmap: str = "RdBu_r",
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (3.2, 3.0),
    title: str = "Slices",
) -> Tuple["plt.Figure", np.ndarray]:
    """
    Small-multiples panel of 1-D or 2-D slices taken at evenly spaced
    indices along `sweep_axis`, with all other non-kept axes fixed
    (default: middle index).

    Useful for e.g. viewing a 3-D volume as a grid of z-slices, or a
    4-D (x, y, z, t) field as a grid of time snapshots at fixed z.
    """
    _apply_style()
    if sweep_axis in keep_axes:
        raise ValueError("sweep_axis must not be one of keep_axes")

    specs = _make_axis_specs(field.ndim, field.shape, axis_names, coords)
    n_sweep = field.shape[sweep_axis]
    panel_idxs = np.unique(np.linspace(0, n_sweep - 1, n_panels).astype(int))
    n_panels = len(panel_idxs)
    nrows = int(np.ceil(n_panels / ncols))

    is_2d = len(keep_axes) == 2
    global_vmax = None
    if is_2d:
        all_vals = [
            slice_field(field, keep_axes, {**(fixed_indices or {}), sweep_axis: int(i)})
            for i in panel_idxs
        ]
        global_vmax = float(max(np.nanmax(np.abs(v)) for v in all_vals)) or 1.0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for panel_num, idx in enumerate(panel_idxs):
        ax = flat_axes[panel_num]
        data = slice_field(field, keep_axes, {**(fixed_indices or {}), sweep_axis: int(idx)})
        sweep_val = specs[sweep_axis].coords[idx]

        if is_2d:
            ax0, ax1 = specs[keep_axes[0]], specs[keep_axes[1]]
            norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0.0, vmax=global_vmax)
            ax.pcolormesh(ax1.coords, ax0.coords, data, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
        else:
            ax0 = specs[keep_axes[0]]
            ax.plot(ax0.coords, data, color="#1f77b4", linewidth=1.6)
            ax.grid(True, alpha=0.3)

        ax.set_title(f"{specs[sweep_axis].name} = {sweep_val:.3g}", fontsize=10)

    for panel_num in range(n_panels, len(flat_axes)):
        flat_axes[panel_num].axis("off")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. 3-D isosurface / volume rendering
# ---------------------------------------------------------------------------

def plot_isosurface_3d(
    field: np.ndarray,
    *,
    coords: Optional[Sequence[np.ndarray]] = None,
    axis_names: Sequence[str] = ("x", "y", "z"),
    mode: str = "isosurface",
    isomin: Optional[float] = None,
    isomax: Optional[float] = None,
    surface_count: int = 3,
    opacity: float = 0.5,
    colorscale: str = "RdBu",
    title: str = "3-D Field",
):
    """
    Render a 3-D scalar field as an interactive isosurface or volume
    plot (requires plotly). Falls back to a matplotlib voxel plot at
    reduced resolution if plotly is not available.

    Parameters
    ----------
    field : array of shape (Nx, Ny, Nz)
    mode  : "isosurface" or "volume"

    Returns
    -------
    A plotly Figure if plotly is available, else a matplotlib
    (fig, ax) tuple.
    """
    if field.ndim != 3:
        raise ValueError(f"plot_isosurface_3d expects a 3-D field, got ndim={field.ndim}")

    if coords is None:
        coords = [np.arange(n) for n in field.shape]

    if _PLOTLY_AVAILABLE:
        X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        vmin = isomin if isomin is not None else float(np.nanpercentile(field, 15))
        vmax = isomax if isomax is not None else float(np.nanpercentile(field, 85))

        if mode == "isosurface":
            trace = go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=field.flatten(),
                isomin=vmin, isomax=vmax,
                surface_count=surface_count,
                opacity=opacity,
                colorscale=colorscale,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        elif mode == "volume":
            trace = go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=field.flatten(),
                isomin=vmin, isomax=vmax,
                opacity=0.1,
                surface_count=17,
                colorscale=colorscale,
            )
        else:
            raise ValueError("mode must be 'isosurface' or 'volume'")

        fig = go.Figure(data=trace)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=axis_names[0],
                yaxis_title=axis_names[1],
                zaxis_title=axis_names[2],
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

    # ---- matplotlib fallback: coarse voxel plot ----
    warnings.warn(
        "plotly not installed; falling back to a coarse matplotlib voxel "
        "plot. Install plotly for proper isosurface/volume rendering: "
        "pip install plotly"
    )
    _apply_style()
    max_res = 24
    step = tuple(max(1, n // max_res) for n in field.shape)
    small = field[::step[0], ::step[1], ::step[2]]
    thresh = isomax if isomax is not None else float(np.nanpercentile(small, 80))
    filled = small >= thresh

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(small.min()), vmax=float(small.max()))
    colors = plt.cm.get_cmap(colorscale if colorscale in plt.colormaps() else "RdBu_r")(norm(small))
    ax.voxels(filled, facecolors=colors, edgecolor=None, alpha=opacity)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Animate an N-D field by sweeping one axis
# ---------------------------------------------------------------------------

def animate_nd_field(
    field: np.ndarray,
    keep_axes: Sequence[int],
    sweep_axis: int,
    *,
    mode: str = "slice",
    reduction: str = "mean",
    fixed_indices: Optional[Dict[int, int]] = None,
    axis_names: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    cmap: str = "RdBu_r",
    title: str = "PhysAI — N-D field evolution",
    figsize: Optional[Tuple[float, float]] = None,
    fps: int = 20,
    interval_ms: int = 50,
    save_path: Optional[str] = None,
) -> "FuncAnimation":
    """
    Animate a 1-D or 2-D view of an N-D field as ``sweep_axis`` (e.g.
    time, or a spectral/parameter axis) advances. All other non-kept
    axes are held fixed (mode="slice") or reduced (mode="project"),
    exactly as in ``plot_slice``.

    Parameters
    ----------
    field      : N-D array
    keep_axes  : 1 or 2 axes to visualise (spatial axes, typically)
    sweep_axis : axis to animate over (must not be in keep_axes)
    save_path  : if given, save as GIF (requires pillow)
    """
    _apply_style()
    if sweep_axis in keep_axes:
        raise ValueError("sweep_axis must not be one of keep_axes")
    if len(keep_axes) not in (1, 2):
        raise ValueError("keep_axes must have length 1 or 2")

    specs = _make_axis_specs(field.ndim, field.shape, axis_names, coords)
    n_frames = field.shape[sweep_axis]
    fixed_indices = dict(fixed_indices or {})

    if mode == "slice":
        def _frame_data(i: int) -> np.ndarray:
            idx = {**fixed_indices, sweep_axis: i}
            return slice_field(field, keep_axes, idx)

    elif mode == "project":
        # Reduce every axis except keep_axes and sweep_axis once up front,
        # then index the sweep axis directly per frame (cheap + correct).
        other_axes = [a for a in range(field.ndim) if a not in keep_axes and a != sweep_axis]
        reduced_field = _REDUCERS[reduction](field, tuple(other_axes)) if other_axes else field
        remaining = [a for a in range(field.ndim) if a in keep_axes or a == sweep_axis]
        target_order = list(keep_axes) + [sweep_axis]
        perm = [remaining.index(a) for a in target_order]
        reduced_field = np.transpose(reduced_field, perm)

        def _frame_data(i: int) -> np.ndarray:
            return reduced_field[..., i]

    else:
        raise ValueError("mode must be 'slice' or 'project'")

    all_frames = [_frame_data(i) for i in range(n_frames)]
    vmax = float(max(np.nanmax(np.abs(f)) for f in all_frames)) or 1.0

    if len(keep_axes) == 1:
        ax0 = specs[keep_axes[0]]
        figsize = figsize or (8, 4)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(float(ax0.coords.min()), float(ax0.coords.max()))
        ax.set_ylim(-vmax * 1.1, vmax * 1.1)
        ax.set_xlabel(ax0.label())
        ax.set_ylabel("field value")
        ax.grid(True, alpha=0.35)
        (line,) = ax.plot([], [], color="#1f77b4", linewidth=2.0)

        def _init():
            line.set_data([], [])
            return (line,)

        def _update(i: int):
            line.set_data(ax0.coords, all_frames[i])
            sweep_val = specs[sweep_axis].coords[i]
            ax.set_title(f"{title}   [{specs[sweep_axis].name} = {sweep_val:.4g}]")
            return (line,)

        anim = FuncAnimation(fig, _update, frames=n_frames, init_func=_init,
                              interval=interval_ms, blit=True)

    else:
        ax0, ax1 = specs[keep_axes[0]], specs[keep_axes[1]]
        figsize = figsize or (7, 5.5)
        fig, ax = plt.subplots(figsize=figsize)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.pcolormesh(ax1.coords, ax0.coords, all_frames[0], cmap=cmap,
                            norm=norm, shading="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im, cax=cax, label="field value")
        ax.set_xlabel(ax1.label())
        ax.set_ylabel(ax0.label())
        ax.set_aspect("equal")

        def _update(i: int):
            im.set_array(all_frames[i].ravel())
            sweep_val = specs[sweep_axis].coords[i]
            ax.set_title(f"{title}   [{specs[sweep_axis].name} = {sweep_val:.4g}]")
            return (im,)

        anim = FuncAnimation(fig, _update, frames=n_frames,
                              interval=interval_ms, blit=False)

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=fps))

    return anim


# ---------------------------------------------------------------------------
# 7. Animate a rotating / evolving 3-D isosurface (4-D field: x,y,z,t)
# ---------------------------------------------------------------------------

def animate_isosurface_3d(
    field_4d: np.ndarray,
    *,
    coords: Optional[Sequence[np.ndarray]] = None,
    axis_names: Sequence[str] = ("x", "y", "z", "t"),
    isomin: Optional[float] = None,
    isomax: Optional[float] = None,
    surface_count: int = 3,
    opacity: float = 0.5,
    colorscale: str = "RdBu",
    title: str = "3-D Field Evolution",
):
    """
    Animate a 4-D field (3 spatial dims + 1 time/parameter dim) as a
    plotly isosurface animation with a play button and time slider.

    Parameters
    ----------
    field_4d : array of shape (Nx, Ny, Nz, Nt)

    Returns
    -------
    A plotly Figure with animation frames (requires plotly).
    """
    _require_plotly()
    if field_4d.ndim != 4:
        raise ValueError(f"expected a 4-D field (x, y, z, t), got ndim={field_4d.ndim}")

    if coords is None:
        coords = [np.arange(n) for n in field_4d.shape]
    x, y, z, t = coords
    Nt = field_4d.shape[3]

    vmin = isomin if isomin is not None else float(np.nanpercentile(field_4d, 15))
    vmax = isomax if isomax is not None else float(np.nanpercentile(field_4d, 85))

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    def _iso_trace(vals: np.ndarray) -> "go.Isosurface":
        return go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=vals.flatten(),
            isomin=vmin, isomax=vmax,
            surface_count=surface_count,
            opacity=opacity,
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

    frames = [
        go.Frame(data=[_iso_trace(field_4d[..., k])], name=f"{t[k]:.4g}")
        for k in range(Nt)
    ]

    fig = go.Figure(data=[_iso_trace(field_4d[..., 0])], frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2],
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 80, "redraw": True},
                                   "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            steps=[
                dict(method="animate",
                     args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                       "mode": "immediate"}],
                     label=f.name)
                for f in frames
            ],
            currentvalue={"prefix": f"{axis_names[3]} = "},
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Optional interactive Jupyter explorer
# ---------------------------------------------------------------------------

def interactive_nd_explorer(
    field: np.ndarray,
    keep_axes: Sequence[int] = (0, 1),
    *,
    axis_names: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    cmap: str = "RdBu_r",
):
    """
    Jupyter widget for interactively slicing through an N-D field.
    Sliders are created for every axis not in ``keep_axes``.

    Requires ``ipywidgets``. Falls back to a static plot with a
    warning if unavailable.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        warnings.warn(
            "ipywidgets not installed; showing a single static slice "
            "instead of an interactive explorer. "
            "Install with: pip install ipywidgets"
        )
        return plot_slice(field, keep_axes, axis_names=axis_names, coords=coords, cmap=cmap)

    _apply_style()
    specs = _make_axis_specs(field.ndim, field.shape, axis_names, coords)
    other_axes = [a for a in range(field.ndim) if a not in keep_axes]

    sliders = {
        a: widgets.IntSlider(
            value=field.shape[a] // 2, min=0, max=field.shape[a] - 1,
            description=specs[a].name, continuous_update=False,
        )
        for a in other_axes
    }

    out = widgets.Output()

    def _redraw(**kwargs):
        with out:
            out.clear_output(wait=True)
            fixed = {a: kwargs[specs[a].name] for a in other_axes}
            fig, _ = plot_slice(
                field, keep_axes, mode="slice", fixed_indices=fixed,
                axis_names=axis_names, coords=coords, cmap=cmap,
            )
            plt.show()

    ui = widgets.interactive_output(
        _redraw, {specs[a].name: sliders[a] for a in other_axes}
    )
    display(widgets.VBox(list(sliders.values())), ui)
    return ui


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "AxisSpec",
    "describe_field",
    "slice_field",
    "project_field",
    "plot_slice",
    "plot_slice_grid",
    "plot_isosurface_3d",
    "animate_nd_field",
    "animate_isosurface_3d",
    "interactive_nd_explorer",
]