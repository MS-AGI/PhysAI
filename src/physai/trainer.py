"""
physai/trainer.py

Unified, backend-agnostic training engine for PhysAI.

Responsibilities
----------------
* Consumes a RuntimeConfig emitted by AutoOptimizer (or a manually-built one).
* Orchestrates the full training loop: Adam warm-up → optional L-BFGS fine-tune.
* Implements Residual-Adaptive Refinement (RAR) collocation resampling.
* Supports learning-rate scheduling (cosine, step, exponential).
* Gradient clipping (backend-agnostic).
* Checkpoint save/restore (numpy-serialised, framework-independent).
* Structured history logging with per-term loss tracking.
* TensorFlow GradientTape context management is handled transparently.
* JAX functional-update pattern is encapsulated behind a common step API.

Design constraints
------------------
* Zero .detach() calls — graphs stay alive for higher-order AD.
* No Python control flow over tensor values (no if tensor > 0).
* Every mutation path (param updates, LR changes) goes through backend
  primitives so the Trainer is backend-agnostic up to the optimizer_step shim.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from physai.backends.base import AbstractBackend, Tensor
from physai.core.auto_optimizer import (
    AutoOptimizer,
    DomainSpec,
    RuntimeConfig,
    _uniform_sample,
    _no_grad_context,
)
from physai.core.losses import (
    WeightedLossComposite,
    dirichlet_loss,
    neumann_loss,
    robin_loss,
    periodic_loss,
)
from physai.core.pde_residual import build_residual, MixedResidual, PDEResidual
from physai.geometry import BoundaryConditionSet, Geometry

# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Structured record of every logged training step."""
    steps:       List[int]            = field(default_factory=list)
    total_loss:  List[float]          = field(default_factory=list)
    terms:       Dict[str, List[float]] = field(default_factory=dict)
    lr:          List[float]          = field(default_factory=list)
    wall_time:   List[float]          = field(default_factory=list)
    # Populated if validation points provided
    val_loss:    List[float]          = field(default_factory=list)

    def record(
        self,
        step: int,
        total: float,
        breakdown: Dict[str, float],
        lr: float,
        t: float,
        val: Optional[float] = None,
    ) -> None:
        self.steps.append(step)
        self.total_loss.append(total)
        self.lr.append(lr)
        self.wall_time.append(t)
        for k, v in breakdown.items():
            self.terms.setdefault(k, []).append(v)
        if val is not None:
            self.val_loss.append(val)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps":      self.steps,
            "total_loss": self.total_loss,
            "terms":      self.terms,
            "lr":         self.lr,
            "wall_time":  self.wall_time,
            "val_loss":   self.val_loss,
        }

    def save_npz(self, path: str) -> None:
        arrays: Dict[str, np.ndarray] = {
            "steps":      np.array(self.steps),
            "total_loss": np.array(self.total_loss),
            "lr":         np.array(self.lr),
            "wall_time":  np.array(self.wall_time),
        }
        for k, v in self.terms.items():
            arrays[f"term_{k}"] = np.array(v)
        if self.val_loss:
            arrays["val_loss"] = np.array(self.val_loss)
        np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

class Callback:
    """
    Base callback. Override any hook you need.
    All hooks receive the Trainer instance so they can inspect any state.
    """
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "Trainer",
        step: int,
        loss: float,
        breakdown: Dict[str, float],
    ) -> None:
        pass

    def on_lbfgs_begin(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    def on_rar_step(self, trainer: "Trainer", step: int) -> None:
        pass


class EarlyStoppingCallback(Callback):
    """Stop training when total loss does not improve by ``min_delta`` for
    ``patience`` consecutive log intervals."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-8) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = float("inf")
        self._no_improve = 0

    def on_epoch_end(self, trainer, step, loss, breakdown):
        if self._best - loss > self.min_delta:
            self._best       = loss
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            trainer._stop_flag = True


class CheckpointCallback(Callback):
    """Save model parameters to disk every ``every`` steps."""

    def __init__(self, directory: str, every: int = 1000) -> None:
        self.directory = Path(directory)
        self.every     = every
        self.directory.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer, step, loss, breakdown):
        if step % self.every == 0:
            trainer.save_checkpoint(str(self.directory / f"ckpt_step{step:07d}"))


class PrintCallback(Callback):
    """Formatted console output at each log step."""

    def __init__(self, every: int = 100) -> None:
        self.every = every
        self._t0   = time.perf_counter()

    def on_train_begin(self, trainer):
        self._t0 = time.perf_counter()
        print(f"\n{'─'*70}")
        print(f"  PhysAI Trainer  |  backend={trainer.backend.name}  "
              f"|  device={trainer.backend.default_device()}")
        print(f"{'─'*70}")

    def on_epoch_end(self, trainer, step, loss, breakdown):
        if step % self.every == 0:
            elapsed  = time.perf_counter() - self._t0
            terms_str = "  ".join(
                f"{k}={v:.3e}" for k, v in breakdown.items()
            )
            lr_now = trainer.history.lr[-1] if trainer.history.lr else 0.0
            print(
                f"  step {step:>7d}  |  loss={loss:.4e}  |  "
                f"lr={lr_now:.2e}  |  {terms_str}  |  {elapsed:.1f}s"
            )

    def on_lbfgs_begin(self, trainer):
        print(f"\n{'─'*70}")
        print("  Switching to L-BFGS fine-tuning phase …")
        print(f"{'─'*70}")

    def on_train_end(self, trainer):
        elapsed = time.perf_counter() - self._t0
        best    = min(trainer.history.total_loss) if trainer.history.total_loss else float("nan")
        print(f"\n{'─'*70}")
        print(f"  Training complete  |  best_loss={best:.4e}  |  elapsed={elapsed:.1f}s")
        print(f"{'─'*70}\n")


# ---------------------------------------------------------------------------
# Backend-specific gradient-clip shim
# ---------------------------------------------------------------------------

def _clip_gradients(
    backend: AbstractBackend,
    params: List[Tensor],
    max_norm: float,
) -> None:
    """
    In-place gradient clipping for PyTorch (only backend that accumulates
    gradients on the parameter objects themselves).
    JAX / TF handle gradients functionally; clipping is done before the
    optimizer update step in those paths.
    """
    if backend.name in ("torch", "paddle"):
        # Both are eager frameworks that accumulate gradients directly
        # on the parameter objects (unlike JAX/TF's functional grad
        # pytrees), so both clip the same way.
        if backend.name == "torch":
            import torch
            torch.nn.utils.clip_grad_norm_(params, max_norm)
        else:
            import paddle
            paddle.nn.utils.clip_grad_norm_(params, max_norm)


def _clip_grad_dict(
    backend: AbstractBackend,
    grads: Any,
    max_norm: float,
) -> Any:
    """
    Clip a gradient pytree by global norm. Used for JAX and TensorFlow
    functional update paths.

    JAX (Flax-style) params/grads are a genuinely *nested* pytree —
    typically ``{"Dense_0": {"bias": ..., "kernel": ...}, "Dense_1": {...},
    ...}`` — not a flat dict of leaf tensors. A single level of
    ``dict.values()`` flattening only reaches the per-layer sub-dicts, not
    the actual tensors inside them, so ``backend.square()`` was being
    called on a ``dict`` and crashing. JAX gets its own branch using
    ``jax.tree_util.tree_leaves``/``tree_map``, which flatten/rebuild a
    pytree of arbitrary nesting depth correctly. TensorFlow's path passes
    a flat list (one gradient per variable, since
    ``tape.gradient(total, self.model.parameters())`` already returns a
    flat list matching the flat ``self.model.parameters()`` list) so the
    simpler single-level logic below is correct and sufficient for it.
    """
    b = backend

    if b.name == "jax":
        import jax
        leaves = jax.tree_util.tree_leaves(grads)
        total_sq = sum(
            float(b.to_numpy(b.sum(b.square(g))))
            for g in leaves if g is not None
        )
        global_norm = float(np.sqrt(total_sq + 1e-12))
        clip_coeff  = min(max_norm / (global_norm + 1e-6), 1.0)
        return jax.tree_util.tree_map(lambda g: g * clip_coeff, grads)

    # Flatten to list of tensors
    if isinstance(grads, (list, tuple)):
        flat = list(grads)
    elif isinstance(grads, dict):
        flat = list(grads.values())
    else:
        flat = [grads]

    total_sq = sum(
        float(b.to_numpy(b.sum(b.square(g))))
        for g in flat if g is not None
    )
    global_norm = float(np.sqrt(total_sq + 1e-12))
    clip_coeff  = min(max_norm / (global_norm + 1e-6), 1.0)

    if isinstance(grads, (list, tuple)):
        return type(grads)(g * clip_coeff if g is not None else g for g in grads)
    if isinstance(grads, dict):
        return {k: (v * clip_coeff if v is not None else v) for k, v in grads.items()}
    return grads * clip_coeff


# ---------------------------------------------------------------------------
# LR schedule application (backend-agnostic)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Unified physics-informed training engine.

    Usage
    -----
    ::

        backend = get_backend("torch")
        config  = AutoOptimizer(backend).analyse(spec)

        pinn    = build_pinn(backend, n_input=2, n_output=1,
                             hidden_sizes=(128,)*4)
        residual = build_residual("burgers", backend, nu=0.01)

        trainer = Trainer(
            backend   = backend,
            config    = config,
            model     = pinn,
            residual  = residual,
            collocation_points = x_coll,
            bc_points          = x_bc,
            bc_values          = u_bc,
        )
        history = trainer.train()

    Parameters
    ----------
    backend            : AbstractBackend
    config             : RuntimeConfig from AutoOptimizer (or built manually)
    model              : PINN or FNO instance
    residual           : PDEResidual instance (or any callable returning residual)
    collocation_points : interior collocation points  [N_coll, d]
    bc_points          : boundary condition points    [N_bc, d]  — simple
                         Dirichlet-only path, kept for backward
                         compatibility with box-shaped domains.
    bc_values          : prescribed BC values         [N_bc, n_out]
    boundary_conditions: a ``physai.geometry.BoundaryConditionSet``
                         describing arbitrary (possibly mixed) Dirichlet /
                         Neumann / Robin / periodic conditions over
                         *any* geometry (box, CSG shape, custom SDF, or
                         an uploaded mesh). When given, the trainer draws
                         boundary points from ``boundary_conditions.
                         geometry.sample_boundary(...)`` itself, tags them
                         by region, and adds one loss term per condition —
                         this supersedes ``bc_points``/``bc_values`` and
                         the two should not both be given non-None.
    n_bc_samples       : number of boundary points to draw when
                         ``boundary_conditions`` is given (defaults to
                         ``config.training.n_bc_points``)
    ic_points          : initial condition points     [N_ic, d]  (optional)
    ic_values          : prescribed IC values         [N_ic, n_out]  (optional)
    data_points        : observation locations        [N_data, d]  (optional)
    data_values        : observation values           [N_data, n_out]  (optional)
    val_points         : validation points            (optional)
    val_values         : validation values            (optional)
    callbacks          : list of Callback instances
    log_every          : log interval in steps
    auto_optimizer     : AutoOptimizer instance for RAR (created if None)

    For ``config.model.arch == "spectral_element"`` (the USENO CP-factored
    model, see ``physai.models.spectral_element`` / ``spectralpinn``), do
    not use this constructor directly — the collocation/BC/IC point-cloud
    interface above doesn't apply to that model at all (it takes a batch
    of scalar time points, not spatial collocation points, and its "loss"
    is a CP-tensor-norm residual plus element-interface continuity terms,
    not a per-point PDE residual). Use ``Trainer.for_spectral_element(...)``
    instead — it builds the right objects via
    ``physai.models.build_spectral_element_trainer`` and reuses this
    class's ``train()`` loop, ``TrainingHistory``, callbacks, and
    checkpointing, but *not* RAR resampling, the validation-loss hook, or
    the L-BFGS fine-tuning phase, none of which have a meaningful
    equivalent for this architecture (see ``for_spectral_element`` and
    ``_train_spectral_element`` docstrings for exactly what is and isn't
    shared).
    """

    def __init__(
        self,
        backend: AbstractBackend,
        config: RuntimeConfig,
        model: Any,
        residual: Any,
        *,
        collocation_points: Tensor,
        bc_points: Optional[Tensor] = None,
        bc_values: Optional[Tensor] = None,
        boundary_conditions: Optional[BoundaryConditionSet] = None,
        n_bc_samples: Optional[int] = None,
        ic_points: Optional[Tensor] = None,
        ic_values: Optional[Tensor] = None,
        data_points: Optional[Tensor] = None,
        data_values: Optional[Tensor] = None,
        val_points: Optional[Tensor] = None,
        val_values: Optional[Tensor] = None,
        callbacks: Optional[List[Callback]] = None,
        log_every: int = 100,
        auto_optimizer: Optional[AutoOptimizer] = None,
        live_dashboard: bool = False,
        dashboard_chat: bool = False,
        dashboard_model_path: Optional[str] = None,
        diff_mode: str = "reverse",
        compile: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        live_dashboard : if True, attach a live ``rich`` terminal dashboard
            (replaces the default ``PrintCallback`` unless ``callbacks`` is
            also given, in which case the dashboard is appended). Requires
            the optional 'rich' dependency: ``pip install physai[dashboard]``.
        dashboard_chat  : if True (and live_dashboard=True), also spin up a
            local llama.cpp chat panel next to the training logs. Requires
            ``pip install physai[chat]`` and ``dashboard_model_path``.
        dashboard_model_path : path to a local GGUF model for the chat panel.
        diff_mode : differentiation strategy ("reverse" | "forward" |
            "taylor") used by the Trainer's *own* derivative helper
            (``_spatial_grad``, used for Neumann/Robin boundary condition
            evaluation). This is independent of any ``diff_mode`` already
            configured on ``residual`` itself — set that on the residual
            instance (e.g. ``build_residual("heat", backend,
            diff_mode="taylor")``) to control the PDE residual's own
            derivative computation. Default "reverse" matches prior
            behavior exactly.
        compile : if True, compile the per-step loss computation
            (``self._composite`` — the actual PDE/BC/IC/data loss
            assembly, called once per training step) via
            ``backend.jit()`` instead of running it eager every step.

            Only meaningfully supported for ``backend.name in {"torch",
            "tensorflow"}`` — both ship a real ahead-of-time/graph
            compiler (``torch.compile``'s TorchInductor, TF's XLA via
            ``jit_compile=True``) that fuses ops and removes per-op
            Python dispatch overhead, which is where most of the
            achievable speedup actually lives for typical PINN/FNO
            training steps (see the design discussion this flag came
            from — an AOT-to-native-exe pipeline was considered and
            rejected because it wouldn't beat this by any meaningful
            margin, since both frameworks' eager mode already just
            dispatches into the same C++ kernels this would use).

            For ``jax``/``paddle`` this flag is accepted but ignored
            with a one-time warning: JAX already traces every call
            under the hood (so an explicit wrap here would mostly
            just fix re-tracing overhead, not add new fusion the way
            TorchInductor/XLA do), and adding real "compile" support
            for those two was out of scope for this change — not
            silently faked as equivalent.

            Compilation only happens lazily on the first training
            step (so ``compile=True`` costs nothing until ``train()``
            actually runs), and only once per ``Trainer`` instance.
        compile_kwargs : extra keyword arguments forwarded to
            ``backend.jit()`` when ``compile=True``. For torch this
            reaches ``torch.compile(func, **compile_kwargs)`` (e.g.
            ``{"mode": "max-autotune"}``). For tensorflow this reaches
            ``tf.function(func, **compile_kwargs)`` — pass
            ``{"jit_compile": True}`` explicitly to actually enable
            XLA (the default plain ``tf.function`` wrap alone gives a
            smaller win than XLA does, so it's opt-in via this dict
            rather than silently assumed).
        """
        self.backend    = backend
        self.config     = config
        self.model      = model
        self.residual   = residual
        self.diff_mode  = diff_mode

        self._compile        = compile
        self._compile_kwargs = compile_kwargs or {}
        self._composite_compiled: Optional[Callable] = None
        self._compile_unsupported_warned = False

        # Collocation points (mutable — updated by RAR)
        self._coll   = collocation_points
        self._bc_pts = bc_points
        self._bc_val = bc_values

        if boundary_conditions is not None and (bc_points is not None or bc_values is not None):
            raise ValueError(
                "Trainer: pass either (bc_points, bc_values) OR "
                "boundary_conditions, not both — they're two different "
                "ways of specifying the same loss terms."
            )
        self._bcs = boundary_conditions
        self._n_bc_samples = n_bc_samples
        self._ic_pts = ic_points
        self._ic_val = ic_values
        self._d_pts  = data_points
        self._d_val  = data_values
        self._v_pts  = val_points
        self._v_val  = val_values

        self.log_every      = log_every

        if live_dashboard:
            # Lazy import: 'rich' (and 'llama_cpp' if chat is requested)
            # stay optional dependencies, only touched when this feature
            # is actually used.
            from physai.dashboard import RichDashboardCallback

            dashboard_cb = RichDashboardCallback(
                total_steps=config.problem.max_epochs,
                enable_chat=dashboard_chat,
                model_path=dashboard_model_path,
            )
            self.callbacks = list(callbacks) + [dashboard_cb] if callbacks else [dashboard_cb]
        else:
            self.callbacks = callbacks or [PrintCallback(log_every)]

        self._stop_flag     = False
        self.history        = TrainingHistory()
        self._step          = 0
        self._t0            = 0.0

        self._auto_opt = auto_optimizer or AutoOptimizer(backend, verbose=False)

        # Build optimizer
        tc     = config.training
        self._optimizer = backend.build_optimizer(
            model          = model._model,
            optimizer_name = tc.optimizer_name,
            lr             = tc.learning_rate,
            weight_decay   = tc.weight_decay if tc.weight_decay > 0 else 0.0,
        )

        # Loss composite
        self._composite = WeightedLossComposite(
            backend  = backend,
            strategy = "fixed",
        )

        # If an arbitrary-geometry BoundaryConditionSet was given, sample
        # its boundary points now (before loss terms are registered, since
        # registration checks which BC data is present).
        self._bc_targets: Dict[str, Dict[str, Any]] = {}
        if self._bcs is not None:
            self.resample_boundary_conditions()

        self._setup_loss_terms()

    # ------------------------------------------------------------------
    # Alternate constructor — "spectral_element" arch (USENO)
    # ------------------------------------------------------------------

    @classmethod
    def for_spectral_element(
        cls,
        config: RuntimeConfig,
        dim_coeffs: List[Dict[int, float]],
        t_points: "torch.Tensor",
        nonlinear_fn: Callable,
        element_widths: Optional[Tensor] = None,
        lr: float = 1e-4,
        callbacks: Optional[List[Callback]] = None,
        log_every: int = 100,
    ) -> "Trainer":
        """
        Build a ``Trainer`` for ``config.model.arch == "spectral_element"``
        (the USENO CP-factored spectral-element model) — the actual
        integration point between that architecture and this class's
        shared ``train()`` loop / ``TrainingHistory`` / callbacks /
        checkpointing, since the normal ``Trainer(...)`` constructor's
        collocation/BC/IC point-cloud interface doesn't apply here (see
        the class docstring).

        Internally this calls ``physai.models.build_spectral_element_trainer``
        (which sizes ``USENOCPModule`` from ``config.model.n_elements`` /
        ``n_modes`` / ``rank``, set by ``AutoOptimizer`` for this arch) and
        wraps the resulting ``USENOPINNTrainer`` so ``train()`` delegates
        each step to it instead of running the generic PDE/BC/IC
        composite-loss machinery.

        Parameters
        ----------
        config       : RuntimeConfig with ``model.arch == "spectral_element"``
        dim_coeffs   : per-axis derivative-order -> coefficient dicts —
                       see ``build_separable_linear_operator``'s docstring,
                       including its documented limitation on where a
                       zeroth-order (reaction) term must go.
        t_points     : ``(batch, 1)`` tensor of scalar time samples fed to
                       ``USENOCPModule`` each step (this model's backbone
                       takes only a time input — see that class's
                       docstring). Re-sampling this per call is the
                       caller's responsibility; unlike RAR for the
                       collocation-point archs, there is no automatic
                       resampling here (see ``_train_spectral_element``).
        nonlinear_fn : ``(spatial_1, spatial_2) -> spatial_result``, the
                       PDE's actual nonlinear term, forwarded to
                       ``USENOCPModule.compute_residuals_and_boundaries``
                       every step (e.g. Burgers' ``u*u`` — pass
                       ``lambda a, b: a * b`` — or a cubic NLS
                       nonlinearity — pass something like
                       ``lambda a, b: a * b * b``). Passing ``None`` here
                       falls back to the plain product, which is only
                       correct if that's actually your PDE's nonlinearity.
        element_widths, lr : forwarded to ``build_spectral_element_trainer``.
        callbacks, log_every : same meaning as the main constructor —
                       ``on_train_begin``/``on_epoch_end``/``on_train_end``
                       all fire normally; ``on_rar_step``/``on_lbfgs_begin``
                       never fire for this arch (see
                       ``_train_spectral_element``).

        What is NOT shared with the generic constructor's instances, and
        why: RAR resampling (there are no collocation points to
        re-sample), the validation-loss hook (no ``val_points``/
        ``val_values`` concept here — plug validation into a custom
        callback if needed), and the L-BFGS fine-tuning phase (USENO
        trains with AdamW + AMP + gradient clipping internally, per the
        reference paper; grafting a second, unrelated optimizer phase on
        top isn't something this integration attempts).
        """
        if config.model.arch != "spectral_element":
            raise ValueError(
                "Trainer.for_spectral_element requires config.model.arch "
                f"== 'spectral_element', got '{config.model.arch}'. Use "
                "the normal Trainer(...) constructor for 'pinn'/'fno'."
            )

        from physai.backends.torch_backend import TorchBackend
        from physai.models import build_spectral_element_trainer

        model, usenopinn = build_spectral_element_trainer(
            config, dim_coeffs, element_widths=element_widths, lr=lr,
        )

        self = cls.__new__(cls)  # bypass __init__: its point-cloud args don't apply
        self.backend   = TorchBackend()  # USENOCPModule is torch-only; needed so
                                          # _apply_lr/save_checkpoint's `b.name`/
                                          # `b.to_numpy` checks keep working unmodified
        self.config    = config
        self.model     = model
        self.residual  = None
        self.diff_mode = "reverse"

        self._coll, self._bc_pts, self._bc_val = None, None, None
        self._bcs, self._n_bc_samples = None, None
        self._ic_pts, self._ic_val = None, None
        self._d_pts, self._d_val = None, None
        self._v_pts, self._v_val = None, None
        self._bc_targets = {}

        self.log_every  = log_every
        self.callbacks  = callbacks or [PrintCallback(log_every)]
        self._stop_flag = False
        self.history    = TrainingHistory()
        self._step      = 0
        self._t0        = 0.0
        self._auto_opt  = AutoOptimizer(self.backend, verbose=False)

        self._optimizer = usenopinn.optimizer  # torch AdamW — reuse _apply_lr as-is
        self._composite = None

        self._useno          = usenopinn
        self._useno_t        = t_points
        self._useno_nonlinear_fn = nonlinear_fn

        return self

    def _train_spectral_element(self, extra_epochs: Optional[int]) -> TrainingHistory:
        """
        ``train()``'s delegate for ``config.model.arch == "spectral_element"``
        — same overall shape (LR schedule, step, log, callbacks, history,
        convergence check) as the generic loop, but each step calls
        ``USENOPINNTrainer.train_step`` instead of the composite PDE/BC/IC
        loss, and skips RAR resampling, the validation-loss hook, and the
        L-BFGS phase (none apply to this architecture — see
        ``for_spectral_element``'s docstring for why).
        """
        max_epochs = (extra_epochs or 0) + self.config.problem.max_epochs
        self._t0 = time.perf_counter()

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for step in range(self._step, max_epochs):
            if self._stop_flag:
                break

            lr_now = self._auto_opt.lr_at_step(self.config, step)
            self._apply_lr(lr_now)

            metrics = self._useno.train_step(self._useno_t, self._useno_nonlinear_fn)
            total_val = metrics["loss_total"]
            bd_val = {k: v for k, v in metrics.items() if k != "loss_total"}

            elapsed = time.perf_counter() - self._t0
            self.history.record(step, total_val, bd_val, lr_now, elapsed, val=None)

            for cb in self.callbacks:
                cb.on_epoch_end(self, step, total_val, bd_val)

            if total_val < self.config.problem.target_loss:
                break

            self._step = step + 1

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history

    # ------------------------------------------------------------------
    # Loss term registration
    # ------------------------------------------------------------------

    def _setup_loss_terms(self) -> None:
        tc = self.config.training
        w  = tc.loss_weights

        self._composite.add(
            "pde",
            lambda: self._pde_loss(),
            weight=w.get("pde", 1.0),
        )
        if self._bc_pts is not None and self._bc_val is not None:
            self._composite.add(
                "bc",
                lambda: self._bc_loss(),
                weight=w.get("bc", 10.0),
            )
        if self._bcs is not None:
            default_w = w.get("bc", 10.0)
            for bc in self._bcs.conditions:
                # closures capture `bc` by value via the default-arg trick
                self._composite.add(
                    bc.name,
                    lambda _bc=bc: self._bc_condition_loss(_bc),
                    weight=w.get(bc.name, default_w),
                )
        if self._ic_pts is not None and self._ic_val is not None:
            self._composite.add(
                "ic",
                lambda: self._ic_loss(),
                weight=w.get("ic", 5.0),
            )
        if self._d_pts is not None and self._d_val is not None:
            self._composite.add(
                "data",
                lambda: self._data_loss(),
                weight=w.get("data", 1.0),
            )

    # ------------------------------------------------------------------
    # Individual loss evaluators
    # ------------------------------------------------------------------

    def _pde_loss(self) -> Tensor:
        """
        Evaluate the PDE loss for one training step.

        Historically this called ``self.residual(model_fn, points)`` and
        assumed a single Tensor back (``mean(square(residual))``). That
        assumption breaks for ``MixedResidual``, whose ``__call__`` returns
        a ``Dict[str, Tensor]`` (one raw residual tensor per coupled field)
        so per-term weighting/logging stays available — squaring a dict
        directly would crash.

        Fix: prefer the residual's own ``.loss()`` method when present —
        every ``PDEResidual`` (including ``MixedResidual``) already defines
        one that does the right thing (plain MSE for single-field
        residuals, weighted sum of per-term MSEs for ``MixedResidual``).
        Fall back to the old call-and-square path only for bare callables
        that aren't ``PDEResidual`` instances (e.g. a user-supplied
        function passed directly as ``residual=``), preserving backward
        compatibility for that case.
        """
        b = self.backend
        if isinstance(self.residual, PDEResidual) or hasattr(self.residual, "loss"):
            return self.residual.loss(self.model.model_fn, self._coll)
        res = self.residual(self.model.model_fn, self._coll)
        return b.mean(b.square(res))

    def _bc_loss(self) -> Tensor:
        b   = self.backend
        u   = self.model.model_fn(self._bc_pts)
        return b.mean(b.square(u - self._bc_val))

    # -- arbitrary-geometry boundary conditions ------------------------

    def resample_boundary_conditions(self) -> None:
        """
        (Re-)draw boundary points from ``self._bcs.geometry`` and tag them
        by condition via ``BoundaryConditionSet.residual_targets``, storing
        backend tensors ready for loss evaluation. Safe to call again later
        (e.g. from a callback) to refresh boundary sampling the same way
        RAR refreshes interior collocation points — useful for curved or
        mesh-based boundaries where a fixed sample can under-cover thin
        features over a long run.
        """
        if self._bcs is None:
            return
        b  = self.backend
        tc = self.config.training
        n  = self._n_bc_samples or tc.n_bc_points

        np_dtype = np.float64 if tc.dtype == "float64" else np.float32
        raw_pts = self._bcs.geometry.sample_boundary(n).astype(np_dtype)
        # residual_targets() (and the BoundaryCondition.region/value
        # callables it drives) only ever operate on *spatial* coordinates —
        # see BoundaryConditionSet.residual_targets / geometry.py — so the
        # spatial-only raw_pts are passed through as-is here.
        targets = self._bcs.residual_targets(raw_pts)

        # For time-dependent problems the model was built for [x..., t]
        # input (n_input = spatial_dims + 1), but every point coming out of
        # residual_targets() above is spatial-only. Append an independently
        # sampled time column per condition below, *after* region/value/
        # normal have already been evaluated on the spatial-only points —
        # normals in particular are a property of the spatial geometry and
        # must not see a time column.
        time_domain = self.config.problem.domain.time_domain
        rng = np.random.default_rng()

        def _pad_time(pts_np: np.ndarray) -> np.ndarray:
            if time_domain is None:
                return pts_np
            t0, t1 = time_domain
            t_col = rng.uniform(t0, t1, size=(len(pts_np), 1)).astype(np_dtype)
            return np.concatenate([pts_np, t_col], axis=1)

        built: Dict[str, Dict[str, Any]] = {}
        for bc in self._bcs.conditions:
            entry = targets[bc.name]
            pts_np = np.atleast_2d(entry["points"]).astype(np_dtype)
            item: Dict[str, Any] = {"kind": entry["kind"], "n": len(pts_np)}
            if len(pts_np) == 0:
                built[bc.name] = item
                continue
            if entry["kind"] in ("neumann", "robin"):
                # Normal vectors are purely spatial — compute before padding.
                normal_np = self._bcs.geometry.normal(pts_np).astype(np_dtype)
            item["points"] = b.tensor(_pad_time(pts_np))
            if entry["kind"] == "periodic":
                pair_np = np.atleast_2d(entry["pair_points"]).astype(np_dtype)
                # Share the same sampled time values as their matching
                # points, since periodic BCs compare u(x_left, t) against
                # u(x_right, t) at the *same* instant, not independent ones.
                if time_domain is None:
                    item["pair_points"] = b.tensor(pair_np)
                else:
                    t_col = b.to_numpy(item["points"])[:, -1:]
                    item["pair_points"] = b.tensor(
                        np.concatenate([pair_np, t_col], axis=1)
                    )
            else:
                target_np = np.asarray(entry["target"], dtype=np_dtype)
                item["target"] = b.tensor(target_np)
                if entry["kind"] in ("neumann", "robin"):
                    item["normal"] = b.tensor(normal_np)
                if entry["kind"] == "robin":
                    item["coeff_a"] = entry["coeff_a"]
                    item["coeff_b"] = entry["coeff_b"]
            built[bc.name] = item

        self._bc_targets = built

    def _spatial_grad(self, points: Tensor) -> Tensor:
        """
        ∇u(points) w.r.t. the *full* model input (spatial dims, and the
        time column if present), via the same
        ``backend.grad(lambda x: backend.sum(fn(x)))`` trick already used
        throughout ``pde_residual.py`` for first derivatives — summing
        over the batch is valid *for reverse-mode* because each sample's
        output only depends on its own row of ``points``, so VJP's
        row-wise structure naturally recovers the per-sample gradient
        unaffected by the sum.

        That trick does not carry over to ``mode="forward"``/``"taylor"``:
        a forward-mode JVP sweep differentiates the *scalar* result of
        ``sum(fn(x))`` directly, which has no per-row structure to
        recover — it collapses the whole batch into one number per input
        dimension, not one gradient row per point. So for those two
        modes this falls back to an explicit per-sample loop instead
        (each single-row call is still shape-correct, since the sum
        trick degenerates to the ordinary single-point gradient when
        the batch has exactly one row). This is only used for BC
        evaluation (Neumann/Robin), not the hot per-step PDE residual
        path, so the extra Python-level loop here is an acceptable
        trade for correctness rather than silently returning a
        collapsed, wrongly-shaped gradient.
        """
        b = self.backend
        fn = self.model.model_fn
        g_fn = b.grad(lambda x: b.sum(fn(x)), mode=self.diff_mode)

        if self.diff_mode == "reverse":
            return g_fn(points)

        n = points.shape[0]
        rows = [g_fn(points[i:i + 1]) for i in range(n)]
        return b.stack(rows, axis=0)

    def _bc_condition_loss(self, bc: Any) -> Tensor:
        """
        Dispatch a single ``BoundaryCondition`` to the matching loss kernel
        in ``physai.core.losses``, using the points/targets/normals cached
        by ``resample_boundary_conditions``. Conditions with zero points
        currently assigned (region matched nothing in this sample) return
        a zero loss rather than erroring, so a thin/rare boundary region
        doesn't crash training on an unlucky sampling draw.
        """
        b = self.backend
        entry = self._bc_targets.get(bc.name)
        if entry is None or entry.get("n", 0) == 0:
            return b.zeros(())

        pts = entry["points"]
        kind = entry["kind"]

        if kind == "dirichlet":
            u = self.model.model_fn(pts)
            return dirichlet_loss(b, u, entry["target"], weight=1.0)

        if kind == "neumann":
            grad_full = self._spatial_grad(pts)
            d = self._bcs.geometry.dim
            grad_spatial = grad_full[..., :d]
            return neumann_loss(
                b, grad_spatial, entry["target"], normal=entry["normal"], weight=1.0,
            )

        if kind == "robin":
            u = self.model.model_fn(pts)
            grad_full = self._spatial_grad(pts)
            d = self._bcs.geometry.dim
            grad_spatial = grad_full[..., :d]
            return robin_loss(
                b, u, grad_spatial,
                alpha=entry["coeff_a"], beta=entry["coeff_b"],
                target=entry["target"], normal=entry["normal"], weight=1.0,
            )

        if kind == "periodic":
            u_left = self.model.model_fn(pts)
            u_right = self.model.model_fn(entry["pair_points"])
            return periodic_loss(b, u_left, u_right, weight=1.0)

        raise ValueError(f"Unknown boundary condition kind '{kind}'.")

    def _ic_loss(self) -> Tensor:
        b   = self.backend
        u   = self.model.model_fn(self._ic_pts)
        return b.mean(b.square(u - self._ic_val))

    def _data_loss(self) -> Tensor:
        b   = self.backend
        u   = self.model.model_fn(self._d_pts)
        return b.mean(b.square(u - self._d_val))

    def _val_loss(self) -> Optional[float]:
        if self._v_pts is None:
            return None
        b = self.backend
        with _no_grad_context(b):
            u   = self.model.model_fn(self._v_pts)
            val = b.mean(b.square(u - self._v_val))
        return float(b.to_numpy(val))

    def _apply_lr(self, lr: float) -> None:
        """
        Set the learning rate of the backend-native optimizer for the
        upcoming step. Uniform across all three backends:

        * torch      : mutate ``param_groups`` in place.
        * tensorflow : assign to the optimizer's ``learning_rate`` variable.
        * jax        : the optimizer is built via
                        ``optax.inject_hyperparams`` (see
                        ``JAXBackend.build_optimizer``), which stores
                        ``learning_rate`` as a live entry inside
                        ``opt_state.hyperparams``. We mutate that entry
                        directly so ``AutoOptimizer.lr_at_step`` schedules
                        (cosine / step / exponential / warm-up) actually
                        take effect on JAX, instead of being silently
                        ignored.
        """
        n = self.backend.name
        if n == "torch":
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr
        elif n == "tensorflow":
            self._optimizer.learning_rate.assign(lr)
        elif n == "paddle":
            self._optimizer.set_lr(lr)
        elif n == "jax":
            state = getattr(self, "_jax_opt_state", None)
            if state is not None and hasattr(state, "hyperparams"):
                state.hyperparams["learning_rate"] = lr
            # If init_jax() hasn't run yet, there's no opt_state to patch;
            # the schedule will simply start applying once training begins.
        else:
            raise RuntimeError(f"Unknown backend '{n}'.")

    # ------------------------------------------------------------------
    # Optional compiled loss path (torch.compile / tf.function XLA)
    # ------------------------------------------------------------------

    def _get_composite_fn(self) -> Callable:
        """
        Returns the callable to use for the per-step loss assembly:
        ``self._composite`` compiled via ``backend.jit()`` if
        ``compile=True`` was requested and the active backend actually
        benefits (torch/tensorflow); the plain eager method otherwise.
        Compiles lazily, once, on first call.
        """
        if not self._compile:
            return self._composite

        name = self.backend.name
        if name not in ("torch", "tensorflow"):
            if not self._compile_unsupported_warned:
                warnings.warn(
                    f"Trainer(compile=True) has no meaningful effect for backend "
                    f"'{name}' — only 'torch' (torch.compile) and 'tensorflow' "
                    f"(tf.function) get a real fusion/AOT benefit here. Ignoring "
                    f"compile=True and running eager.",
                    stacklevel=2,
                )
                self._compile_unsupported_warned = True
            return self._composite

        if self._composite_compiled is None:
            compiled_fn = self.backend.jit(self._composite, **self._compile_kwargs)

            def _compiled_with_fallback(*args, **kwargs):
                # torch.compile/tf.function both compile *lazily*, on
                # first call — so a compile-environment failure (e.g. the
                # GPU is too old for Triton, as torch._inductor.exc.
                # GPUTooOldForTriton reports) only surfaces here, not at
                # the `backend.jit(...)` call above. Only the very first
                # invocation gets this safety net: if it fails before
                # ``self._compiled_call_succeeded_once`` is set, that's
                # treated as "this environment can't actually run
                # compiled mode" and permanently falls back to eager, the
                # same degrade-gracefully spirit as the backend-name check
                # above. A failure on a LATER call (after at least one
                # successful compiled call) is NOT swallowed — at that
                # point compilation itself works, so the failure is much
                # more likely a real bug in the loss/model, and hiding
                # that would be worse than the crash.
                if getattr(self, "_compiled_call_succeeded_once", False):
                    return compiled_fn(*args, **kwargs)
                try:
                    result = compiled_fn(*args, **kwargs)
                    self._compiled_call_succeeded_once = True
                    return result
                except Exception as exc:  # noqa: BLE001 - see docstring above
                    warnings.warn(
                        f"Trainer(compile=True): the first compiled call failed "
                        f"in this environment ({type(exc).__name__}: {exc}). "
                        f"Falling back to eager execution for the rest of "
                        f"training instead of crashing. Set compile=False to "
                        f"skip this attempt entirely next time.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._compile = False
                    self._composite_compiled = None
                    return self._composite(*args, **kwargs)

            self._composite_compiled = _compiled_with_fallback
        return self._composite_compiled

    # ------------------------------------------------------------------
    # Single training step — PyTorch path
    # ------------------------------------------------------------------

    def _step_torch(self) -> Tuple[float, Dict[str, float]]:
        import torch
        b   = self.backend
        tc  = self.config.training

        b.zero_grad(self.model._model)
        total, breakdown = self._get_composite_fn()()
        total.backward()

        if tc.grad_clip_norm is not None:
            _clip_gradients(b, self.model.parameters(), tc.grad_clip_norm)

        self._optimizer.step()

        total_val = float(total.item())
        bd_val    = {k: float(v.item()) for k, v in breakdown.items()}
        return total_val, bd_val

    # ------------------------------------------------------------------
    # Single training step — PaddlePaddle path
    # ------------------------------------------------------------------

    def _step_paddle(self) -> Tuple[float, Dict[str, float]]:
        """
        PaddlePaddle is eager/imperative like PyTorch (gradients
        accumulate directly on parameter objects, no separate tape
        object), so this mirrors ``_step_torch`` exactly rather than
        the TensorFlow GradientTape pattern.
        """
        b   = self.backend
        tc  = self.config.training

        b.zero_grad(self.model._model)
        total, breakdown = self._get_composite_fn()()
        total.backward()

        if tc.grad_clip_norm is not None:
            _clip_gradients(b, self.model.parameters(), tc.grad_clip_norm)

        self._optimizer.step()
        self._optimizer.clear_grad()

        total_val = float(total.item())
        bd_val    = {k: float(v.item()) for k, v in breakdown.items()}
        return total_val, bd_val

    def _step_tf(self) -> Tuple[float, Dict[str, float]]:
        import tensorflow as tf
        b  = self.backend
        tc = self.config.training

        with tf.GradientTape(persistent=True) as tape:
            for var in self.model.parameters():
                tape.watch(var)
            total, breakdown = self._get_composite_fn()()

        grads = tape.gradient(total, self.model.parameters())
        del tape

        if tc.grad_clip_norm is not None:
            grads = _clip_grad_dict(b, grads, tc.grad_clip_norm)

        self._optimizer.apply_gradients(
            zip(grads, self.model.parameters())
        )

        total_val = float(b.to_numpy(total))
        bd_val    = {k: float(b.to_numpy(v)) for k, v in breakdown.items()}
        return total_val, bd_val

    # ------------------------------------------------------------------
    # Single training step — JAX path
    # ------------------------------------------------------------------

    def _step_jax(self) -> Tuple[float, Dict[str, float]]:
        import jax
        import optax
        b  = self.backend
        tc = self.config.training

        # JAX: params live outside the model; store them on self
        params = getattr(self, "_jax_params", None)
        if params is None:
            raise RuntimeError(
                "JAX training requires self._jax_params to be initialised. "
                "Call trainer.init_jax(dummy_input) before trainer.train()."
            )

        def loss_fn(p: Any) -> Any:
            # Bind params onto the model itself — forward() reads
            # model._init_params, so this MUST go through set_jax_params().
            self.model.set_jax_params(p)
            # record_history=False: this runs *inside* jax.value_and_grad's
            # trace, where every value is a tracer, not a concrete array —
            # WeightedLossComposite's normal history bookkeeping calls
            # backend.to_numpy() on each term's value, which raises
            # TracerArrayConversionError under tracing. History is instead
            # recorded below, from the concrete post-trace `breakdown`.
            total, breakdown = self._composite(record_history=False)
            return total, breakdown

        (total, breakdown), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        self._composite.record_history_from_values(breakdown)

        if tc.grad_clip_norm is not None:
            grads = _clip_grad_dict(b, grads, tc.grad_clip_norm)

        updates, self._jax_opt_state = self._optimizer.update(
            grads, self._jax_opt_state, params
        )
        self._jax_params = optax.apply_updates(params, updates)
        self.model.set_jax_params(self._jax_params)

        total_val = float(np.asarray(total))
        bd_val    = {k: float(np.asarray(v)) for k, v in breakdown.items()}
        return total_val, bd_val

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _single_step(self) -> Tuple[float, Dict[str, float]]:
        n = self.backend.name
        if n == "torch":
            return self._step_torch()
        if n == "tensorflow":
            return self._step_tf()
        if n == "paddle":
            return self._step_paddle()
        if n == "jax":
            return self._step_jax()
        raise RuntimeError(f"Unknown backend '{n}'.")

    # ------------------------------------------------------------------
    # JAX initialisation helper
    # ------------------------------------------------------------------

    def init_jax(self, dummy_input: Tensor) -> None:
        """
        Must be called before ``train()`` when using the JAX backend.
        Initialises Flax model parameters and the optax optimiser state.
        """
        import optax
        self._jax_params    = self.model.init_jax_params(dummy_input)
        self._jax_opt_state = self._optimizer.init(self._jax_params)

    # ------------------------------------------------------------------
    # Fine-tuning phase after the main Adam/optax warm-up.
    #
    # Real, backend-native L-BFGS on every backend:
    #   * torch      : torch.optim.LBFGS (strong-Wolfe line search)
    #   * jax        : optax.lbfgs (optax >= 0.2.0), via the standard
    #                  value_and_grad_from_state + zoom line-search pattern
    #   * tensorflow : tensorflow_probability.optimizer.lbfgs_minimize,
    #                  which requires flattening all trainable variables
    #                  into one vector (tfp's optimizer is backend-generic,
    #                  not Keras-native, so this glue is unavoidable)
    #
    # If the optional dependency for a given backend (optax>=0.2 / tfp)
    # isn't installed, we fall back to a low-LR first-order fine-tuning
    # pass rather than silently skipping the phase.
    # ------------------------------------------------------------------

    def _lbfgs_phase(self) -> None:
        n = self.backend.name
        if n == "torch":
            self._lbfgs_phase_torch()
        elif n == "jax":
            self._lbfgs_phase_jax()
        elif n == "tensorflow":
            self._lbfgs_phase_tf()
        elif n == "paddle":
            self._lbfgs_phase_paddle()
        else:
            raise RuntimeError(f"Unknown backend '{n}'.")

    def _lbfgs_phase_paddle(self) -> None:
        """
        Paddle exposes L-BFGS only as a functional, flatten-all-params
        routine (``paddle.incubate.optimizer.functional.minimize_lbfgs``)
        with a very different calling convention from torch's stateful
        ``torch.optim.LBFGS`` (closure-based) or optax's
        ``value_and_grad_from_state`` pattern — wiring it in blind here,
        untested against this file's actual loss-closure shape, risks
        silently getting the flatten/unflatten step wrong. Rather than
        guess, this follows the same documented policy already used
        elsewhere in this method for a missing/awkward optional
        dependency: fall back to a low-LR first-order fine-tuning pass
        instead of skipping the phase silently.
        """
        warnings.warn(
            "True L-BFGS fine-tuning is not yet wired up for the paddle "
            "backend (paddle.incubate.optimizer.functional.minimize_lbfgs "
            "has a functional, flatten-all-params calling convention "
            "that needs its own tested integration, not a blind port of "
            "the torch/tf paths). Falling back to low-LR fine-tuning.",
            UserWarning,
        )
        self._fine_tune_phase_fallback()

    def _lbfgs_phase_torch(self) -> None:
        import torch
        for cb in self.callbacks:
            cb.on_lbfgs_begin(self)

        tc       = self.config.training
        lbfgs_opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter    = tc.lbfgs_max_iter,
            history_size = 50,
            line_search_fn = "strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs_opt.zero_grad()
            total, breakdown = self._composite()
            total.backward()
            return total

        lbfgs_opt.step(closure)

    def _lbfgs_phase_jax(self) -> None:
        try:
            import optax
        except ImportError:
            warnings.warn(
                "optax is required for JAX L-BFGS. Falling back to "
                "low-LR fine-tuning.",
                UserWarning,
            )
            self._fine_tune_phase_fallback()
            return

        if not hasattr(optax, "lbfgs"):
            warnings.warn(
                "optax.lbfgs requires optax>=0.2.0 (install via "
                "'pip install physai[jax]'). Falling back to low-LR "
                "fine-tuning instead of true L-BFGS.",
                UserWarning,
            )
            self._fine_tune_phase_fallback()
            return

        params = getattr(self, "_jax_params", None)
        if params is None:
            raise RuntimeError(
                "JAX L-BFGS requires trainer.init_jax(dummy_input) to have "
                "been called before train()."
            )

        for cb in self.callbacks:
            cb.on_lbfgs_begin(self)

        tc = self.config.training

        def _loss_only(p: Any) -> Any:
            # Same side-effecting bind-then-evaluate pattern as _step_jax,
            # and the same record_history=False requirement: optax's
            # value_and_grad_from_state traces this function too, so a
            # concrete-value history append here would hit the same
            # TracerArrayConversionError _step_jax's docstring explains.
            self.model.set_jax_params(p)
            total, _breakdown = self._composite(record_history=False)
            return total

        lbfgs_opt   = optax.lbfgs(memory_size=50)
        lbfgs_state = lbfgs_opt.init(params)
        value_and_grad_fun = optax.value_and_grad_from_state(_loss_only)

        for _ in range(tc.lbfgs_max_iter):
            value, grad = value_and_grad_fun(params, state=lbfgs_state)
            updates, lbfgs_state = lbfgs_opt.update(
                grad, lbfgs_state, params,
                value=value, grad=grad, value_fn=_loss_only,
            )
            params = optax.apply_updates(params, updates)
            self.model.set_jax_params(params)

        self._jax_params = params
        # One concrete (non-traced) evaluation to record a final history
        # point for the L-BFGS phase — per-iteration history isn't
        # available here since `_loss_only` only returns a scalar (optax's
        # value_and_grad_from_state doesn't support has_aux breakdown the
        # way jax.value_and_grad does in _step_jax), but the phase's net
        # effect on each term is still worth one data point.
        _total, _breakdown = self._composite()

    def _lbfgs_phase_tf(self) -> None:
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
        except ImportError:
            warnings.warn(
                "tensorflow_probability is required for TensorFlow L-BFGS "
                "(install via 'pip install physai[tensorflow]'). Falling "
                "back to low-LR fine-tuning instead of true L-BFGS.",
                UserWarning,
            )
            self._fine_tune_phase_fallback()
            return

        for cb in self.callbacks:
            cb.on_lbfgs_begin(self)

        tc        = self.config.training
        variables = list(self.model.parameters())
        shapes    = [v.shape for v in variables]
        sizes     = [int(tf.reduce_prod(s)) for s in shapes]

        def _flatten(vars_list: List[Any]) -> Any:
            return tf.concat([tf.reshape(v, [-1]) for v in vars_list], axis=0)

        def _assign_flat(flat: Any) -> None:
            splits = tf.split(flat, sizes)
            for v, s, shp in zip(variables, splits, shapes):
                v.assign(tf.reshape(s, shp))

        def _value_and_gradients(flat_params: Any) -> Tuple[Any, Any]:
            _assign_flat(flat_params)
            with tf.GradientTape() as tape:
                for v in variables:
                    tape.watch(v)
                total, _breakdown = self._composite()
            grads = tape.gradient(total, variables)
            grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, variables)
            ]
            return total, _flatten(grads)

        init_position = _flatten(variables)
        results = tfp.optimizer.lbfgs_minimize(
            _value_and_gradients,
            initial_position=init_position,
            max_iterations=tc.lbfgs_max_iter,
        )
        _assign_flat(results.position)

    def _fine_tune_phase_fallback(self) -> None:
        """
        First-order fine-tuning fallback, used only when the optional
        L-BFGS dependency for the active backend (optax>=0.2 for jax,
        tensorflow_probability for tensorflow) isn't installed. Runs at
        a fixed low learning rate (1% of the scheduled base rate, floored
        at 1e-6) for `lbfgs_max_iter` steps.
        """
        tc = self.config.training
        fine_tune_lr = max(tc.learning_rate * 0.01, 1e-6)
        self._apply_lr(fine_tune_lr)

        for _ in range(tc.lbfgs_max_iter):
            self._single_step()



    # ------------------------------------------------------------------
    # RAR resampling
    # ------------------------------------------------------------------

    def _maybe_rar(self, step: int) -> None:
        tc = self.config.training
        if not tc.rar_enabled:
            return
        if step == 0 or step % tc.rar_interval != 0:
            return
        self._coll = self._auto_opt.rar_resample(
            config         = self.config,
            model_fn       = self.model.model_fn,
            current_points = self._coll,
            backend        = self.backend,
        )
        for cb in self.callbacks:
            cb.on_rar_step(self, step)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        extra_epochs: Optional[int] = None,
    ) -> TrainingHistory:
        """
        Run the full training procedure defined by ``self.config``.

        Parameters
        ----------
        extra_epochs : if provided, run this many additional epochs
                       on top of config.problem.max_epochs (useful for
                       resume-from-checkpoint).

        Returns
        -------
        TrainingHistory
        """
        if self.config.model.arch == "spectral_element":
            return self._train_spectral_element(extra_epochs)

        from physai.chat_setup import ensure_chat_consent
        self.chat_enabled = ensure_chat_consent()

        tc         = self.config.training
        max_epochs = (extra_epochs or 0) + self.config.problem.max_epochs
        self._t0   = time.perf_counter()

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for step in range(self._step, max_epochs):
            if self._stop_flag:
                break

            # Learning-rate schedule
            lr_now = self._auto_opt.lr_at_step(self.config, step)
            self._apply_lr(lr_now)

            # Gradient step
            total_val, bd_val = self._single_step()

            # RAR resampling
            self._maybe_rar(step)

            # Logging
            elapsed = time.perf_counter() - self._t0
            val_loss = None
            if step % self.log_every == 0:
                val_loss = self._val_loss()
            self.history.record(step, total_val, bd_val, lr_now, elapsed, val_loss)

            # Callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(self, step, total_val, bd_val)

            # Convergence
            if total_val < self.config.problem.target_loss:
                break

            self._step = step + 1

        # Optional L-BFGS fine-tune
        if tc.use_lbfgs_phase and not self._stop_flag:
            self._lbfgs_phase()

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save model parameters + training state as a ``.npz`` archive.
        Backend-agnostic: everything is converted to numpy first.

        Keys are ``f"param::{name}"`` where ``name`` comes straight from
        ``self.model.named_parameters()`` — NOT re-prefixed with
        ``"param_"``, because for every non-torch backend
        ``named_parameters()`` already returns auto-generated keys of the
        form ``"param_0"``, ``"param_1"``, ... (see ``PINN.
        named_parameters``); re-prefixing those produced double-prefixed
        keys like ``"param_param_0"`` that ``load_checkpoint``'s TF branch
        then failed to match against its own (differently-derived, single-
        prefixed) key set — silently restoring nothing at all. Using the
        same ``name`` verbatim on both sides, with a separator ("::") that
        can't collide with either backend's own naming convention, is what
        actually keeps save and load in agreement.
        """
        b        = self.backend
        state    = {"_step": np.array(self._step)}
        for name, param in self.model.named_parameters().items():
            state[f"param::{name}"] = b.to_numpy(param)
        np.savez(path + ".npz", **state)

    def load_checkpoint(self, path: str) -> None:
        """
        Restore model parameters from a ``.npz`` checkpoint.
        Only PyTorch in-place copy is supported for now; JAX/TF require
        re-initialisation from the loaded arrays.
        """
        archive    = np.load(path if path.endswith(".npz") else path + ".npz",
                             allow_pickle=False)
        self._step = int(archive["_step"])
        b          = self.backend
        prefix     = "param::"

        if b.name == "torch":
            import torch
            # NOTE: use the model's own `.named_parameters()` (a native
            # nn.Module method every torch model here exposes — whether
            # it's the PINN/FNO wrapper's proxy or, for the
            # "spectral_element" arch, USENOCPModule itself, which has no
            # `._model` attribute at all since it *is* the nn.Module)
            # rather than reaching inside a wrapper-specific `._model`.
            # Both return references to the same underlying nn.Parameter
            # tensors, so `.copy_()` still mutates the real weights either
            # way — this is strictly more general, not a behavior change
            # for existing PINN/FNO checkpoints.
            named = dict(self.model.named_parameters())
            for key, arr in archive.items():
                if key.startswith(prefix):
                    pname = key[len(prefix):]
                    if pname in named:
                        with torch.no_grad():
                            named[pname].copy_(torch.tensor(arr))
        elif b.name == "tensorflow":
            # Rebuild the *same* name -> variable mapping save_checkpoint
            # used (named_parameters() is a pure function of parameter
            # order, so calling it again here reproduces identical keys).
            named = dict(self.model.named_parameters())
            for key, arr in archive.items():
                if key.startswith(prefix):
                    pname = key[len(prefix):]
                    var = named.get(pname)
                    if var is not None:
                        var.assign(arr)
        elif b.name == "paddle":
            # Same name -> variable reconstruction as the tensorflow
            # branch (named_parameters() is a pure function of parameter
            # order). Paddle's in-place mutation API is
            # `.set_value(...)`, not `.assign(...)`/`.copy_()`.
            import paddle
            named = dict(self.model.named_parameters())
            for key, arr in archive.items():
                if key.startswith(prefix):
                    pname = key[len(prefix):]
                    var = named.get(pname)
                    if var is not None:
                        var.set_value(paddle.to_tensor(arr, dtype=var.dtype))
        elif b.name == "jax":
            # Rebuild param pytree from archive. Keys are "param::param_0",
            # "param::param_1", ... — sort by the *numeric* index, not the
            # raw string: lexicographic order puts "param::param_10" before
            # "param::param_2", silently scrambling the pytree unflatten
            # order for any model with 10+ parameters.
            import jax
            import re

            def _index(k: str) -> int:
                m = re.search(r"(\d+)$", k)
                return int(m.group(1)) if m else -1

            items = [(k, arr) for k, arr in archive.items() if k.startswith(prefix)]
            items.sort(key=lambda kv: _index(kv[0]))
            leaves = [arr for _, arr in items]
            treedef = jax.tree_util.tree_structure(self._jax_params)
            self._jax_params = jax.tree_util.tree_unflatten(treedef, leaves)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, points: Tensor) -> Tensor:
        """Evaluate the trained model at ``points`` without gradient tracking."""
        b = self.backend
        with _no_grad_context(b):
            return self.model.model_fn(points)

    # ------------------------------------------------------------------
    # Cross-validation against a real Dedalus spectral solve
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        *,
        geometry: Optional[Geometry] = None,
        # -- box-domain path (geometry=None): real Dedalus --------------
        domain_type: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        variables: Optional[List[str]] = None,
        equations: Optional[List[str]] = None,
        bcs: Optional[List[str]] = None,
        ics: Optional[Dict[str, Callable[..., np.ndarray]]] = None,
        is_complex: bool = False,
        timestepper: str = "RK443",
        stop_time: Optional[float] = None,
        pinn_output_index: Optional[Dict[str, int]] = None,
        # -- arbitrary-geometry path (geometry given): embedded-boundary FD
        pde: Optional[str] = None,
        bc_value: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        pde_params: Optional[Dict[str, float]] = None,
        source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ic: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_steps: int = 1000,
        array_module: str = "numpy",
        model_output_index: int = 0,
        eval_time: Optional[float] = None,
        # -- shared -------------------------------------------------------
        grid_points: Union[int, Sequence[int]] = 64,
        dt: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Cross-validate this Trainer's model against a classical numerical
        solve of the same PDE, run by ``physai.solvers.dedalus.Solver`` —
        the real numerics, not an approximation of them. One entry point,
        dispatched the same way ``Solver.solve`` itself dispatches:

        * ``geometry=None`` (default) -> a real Dedalus box solve
          (``Solver.solve_box``). Requires ``domain_type``, ``bounds``,
          ``variables``, ``equations``, ``bcs``, ``ics``. Only 1D spatial
          problems, because ``Solver.solve_box`` builds a single
          ``CartesianCoordinates`` axis in that regime — this is a
          limitation of the box path, not something this method works
          around.
        * ``geometry=<a physai.geometry.Geometry>`` -> an embedded-
          boundary finite-difference solve on a masked N-D grid
          (``Solver.solve_geometry``), any dimension the geometry has.
          Requires ``pde`` (one of "poisson"/"helmholtz"/"heat") and
          ``bc_value``. See ``Solver.solve_geometry``'s docstring for why
          it's scoped to those PDE classes.

        In both cases: the classical solve runs independently (its own
        timestepper/discretization/BC scheme), the model is evaluated at
        the exact same grid, and the return value is the L2 error between
        them. Nothing about training is touched by calling this.

        Parameters
        ----------
        geometry     : selects the path, as above.
        domain_type, bounds, variables, equations, bcs, ics, is_complex,
        timestepper  : forwarded to ``Solver.solve_box`` (geometry=None only).
        stop_time    : evaluation time. Defaults to
            ``self.config.problem.domain.time_domain[1]`` if time-dependent,
            else 1.0 (box path) / ``n_steps * dt`` for a "heat" geometry
            solve / omitted entirely for a steady geometry solve.
        pinn_output_index : (box path) maps each Dedalus ``variables``
            name to the model's output column for that field. Defaults to
            positional order.
        pde, bc_value, pde_params, source, ic, n_steps, array_module :
            forwarded to ``Solver.solve_geometry`` (geometry given only).
        model_output_index : (geometry path) which model output column is
            being cross-validated. Default 0 (a single-output model).
        grid_points  : shared by both paths — resolution of the classical
            grid, scalar or per-axis.
        dt           : shared by both paths — timestep of the classical
            solve.

        Returns
        -------
        Dict of L2 errors. Box path: ``{"<var>_l2_abs": ..., "<var>_l2_rel": ...}``
        per Dedalus variable. Geometry path: ``{"l2_abs": ..., "l2_rel": ...,
        "n_compared": ...}``.
        """
        from physai.solvers.dedalus import Solver  # lazy: dedalus/cupy stay optional

        if geometry is not None:
            if pde is None or bc_value is None:
                raise ValueError(
                    "cross_validate(geometry=...) requires 'pde' and 'bc_value'."
                )
            points, mask, fd_values = Solver.solve_geometry(
                geometry,
                pde=pde,
                bc_value=bc_value,
                grid_points=grid_points,
                pde_params=pde_params,
                source=source,
                ic=ic,
                dt=dt,
                n_steps=n_steps,
                array_module=array_module,
            )

            query_points = points[mask]
            time_domain = self.config.problem.domain.time_domain
            if eval_time is None:
                if pde == "heat":
                    eval_time = n_steps * dt
                elif time_domain is not None:
                    eval_time = time_domain[1]

            if eval_time is not None:
                t_col = np.full((query_points.shape[0], 1), eval_time)
                query_np = np.concatenate([query_points, t_col], axis=1)
            else:
                query_np = query_points

            query = self.backend.tensor(query_np)
            pred_np = np.asarray(self.backend.to_numpy(self.model.model_fn(query)))
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(-1, 1)
            pred_vals = pred_np[:, model_output_index].reshape(-1)

            classical_vals = fd_values[mask].reshape(-1)
            abs_err = float(np.linalg.norm(pred_vals - classical_vals))
            rel_err = abs_err / (float(np.linalg.norm(classical_vals)) + 1e-12)
            return {"l2_abs": abs_err, "l2_rel": rel_err, "n_compared": int(mask.sum())}

        # -- box-domain / real-Dedalus path ---------------------------------
        missing = [
            name for name, val in [
                ("domain_type", domain_type), ("bounds", bounds),
                ("variables", variables), ("equations", equations),
                ("bcs", bcs), ("ics", ics),
            ] if val is None
        ]
        if missing:
            raise ValueError(
                f"cross_validate(geometry=None) requires {missing} "
                "(the box/Dedalus path) — or pass `geometry=` for the "
                "arbitrary-geometry path instead."
            )
        if self.config.problem.domain.spatial_dims != 1:
            raise ValueError(
                "cross_validate(geometry=None) only supports 1D spatial "
                f"problems — this Trainer's domain has spatial_dims="
                f"{self.config.problem.domain.spatial_dims}. Pass "
                "`geometry=` (any dimension) instead, if the PDE is one "
                "Solver.solve_geometry supports."
            )

        time_domain = self.config.problem.domain.time_domain
        if stop_time is None:
            stop_time = time_domain[1] if time_domain is not None else 1.0

        x_grid, classical_profiles = Solver.solve_box(
            domain_type=domain_type,
            bounds=bounds,
            grid_points=grid_points,
            variables=variables,
            equations=equations,
            bcs=bcs,
            ics=ics,
            is_complex=is_complex,
            timestepper=timestepper,
            dt=dt,
            stop_time=stop_time,
        )

        x_grid = np.asarray(x_grid).reshape(-1, 1)
        if time_domain is not None:
            t_col = np.full_like(x_grid, stop_time)
            query_np = np.concatenate([x_grid, t_col], axis=1)
        else:
            query_np = x_grid

        query = self.backend.tensor(query_np)
        pred_np = np.asarray(self.backend.to_numpy(self.model.model_fn(query)))
        if pred_np.ndim == 1:
            pred_np = pred_np.reshape(-1, 1)

        errors: Dict[str, float] = {}
        for i, var in enumerate(variables):
            col = pinn_output_index[var] if pinn_output_index is not None else i
            classical_vals = np.asarray(classical_profiles[var]).reshape(-1)
            pred_vals = pred_np[:, col].reshape(-1)
            if pred_vals.shape != classical_vals.shape:
                raise ValueError(
                    f"cross_validate: model output column {col} for "
                    f"variable '{var}' has shape {pred_vals.shape}, but "
                    f"Dedalus's grid has shape {classical_vals.shape}. They "
                    "must match (same grid_points) to compare pointwise."
                )
            abs_err = float(np.linalg.norm(pred_vals - classical_vals))
            rel_err = abs_err / (float(np.linalg.norm(classical_vals)) + 1e-12)
            errors[f"{var}_l2_abs"] = abs_err
            errors[f"{var}_l2_rel"] = rel_err

        return errors

    def __repr__(self) -> str:
        tc = self.config.training
        return (
            f"Trainer(backend={self.backend.name}, "
            f"pde={self.config.problem.pde_name}, "
            f"step={self._step}/{self.config.problem.max_epochs}, "
            f"optimizer={tc.optimizer_name}, lr={tc.learning_rate:.2e})"
        )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "Trainer",
    "TrainingHistory",
    "Callback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "PrintCallback",
]