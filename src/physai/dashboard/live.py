"""
physai/dashboard/live.py

Live terminal dashboard for PhysAI training runs, wired in as a proper
``Callback`` (see ``physai.trainer.Callback``) rather than a standalone
script. Attach it to any ``Trainer`` on any backend and you get:

  * A live-updating panel of training logs (loss, lr, elapsed time),
    driven directly by ``on_epoch_end`` — no polling, no coupling to a
    specific backend.
  * An optional side chat panel backed by a local GGUF model via
    ``llama_cpp``. In chat mode, enter prompts at the terminal while
    training runs; model responses are generated on a worker thread so
    they do not block training.

Both ``rich`` and ``llama_cpp`` are imported lazily, inside
``on_train_begin``, so importing
``physai.dashboard`` (or ``physai`` as a whole) never requires either
package unless you actually instantiate this callback with the chat
feature enabled.

At training end, the callback writes ``physai_training_report.pdf`` by
default, with total and per-term loss curves and a target-loss convergence
assessment. Set ``report_file=None`` to disable the PDF export.

Usage
-----
The built-in way — ``Trainer`` will construct and attach this callback
for you, so you never need to import ``physai.dashboard`` directly::

    trainer = Trainer(
        backend=backend, config=config, model=model, residual=residual,
        collocation_points=coll,
        live_dashboard=True,               # dashboard on, no chat
    )
    trainer.train()

With the embedded local chat panel::

    trainer = Trainer(
        ..., live_dashboard=True, dashboard_chat=True,
        dashboard_model_path="Llama-3-8B-Instruct.Q4_K_M.gguf",
    )
    trainer.train()  # chat runs alongside training; returns when training ends

Manual attachment is still supported for custom callback stacks
(e.g. combining it with ``EarlyStoppingCallback``)::

    from physai.dashboard import RichDashboardCallback

    dashboard = RichDashboardCallback(total_steps=config.problem.max_epochs)
    trainer = Trainer(..., callbacks=[dashboard, EarlyStoppingCallback()])

Either way, ``rich`` (and ``llama-cpp-python`` for chat) are only ever
imported when the dashboard is actually requested — plain ``import
physai`` never needs them.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from physai.trainer import Callback

__all__ = ["RichDashboardCallback"]

_DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant embedded in a live PhysAI training dashboard. "
    "You can see the training logs shown alongside this chat. Answer "
    "questions about the run concisely and do not invent metrics you "
    "have not been shown."
)


class RichDashboardCallback(Callback):
    """
    Live ``rich``-based terminal dashboard, driven by Trainer hooks.

    Parameters
    ----------
    total_steps   : total number of training steps, for progress display.
    log_maxlen    : number of most-recent log lines kept on screen.
    refresh_hz    : screen refresh rate.
    enable_chat   : if True, spin up a local llama.cpp model and a side
                    chat panel. Requires the ``llama-cpp-python`` package
                    and a local GGUF file at ``model_path``.
    model_path    : path to a local GGUF model file (only used if
                    ``enable_chat=True``).
    n_ctx         : context window for the llama.cpp model.
    system_prompt : optional system prompt for the embedded chat model.
    log_file      : if given, session logs + chat history are written
                    here at the end of training.
    report_file   : if given, a PDF summary with loss plots and a
                    target-convergence assessment is written at the end.
    """

    def __init__(
        self,
        total_steps: Optional[int] = None,
        *,
        log_maxlen: int = 100,
        refresh_hz: int = 4,
        enable_chat: bool = False,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        log_file: Optional[str] = "physai_session.log",
        report_file: Optional[str] = "physai_training_report.pdf",
    ) -> None:
        if enable_chat and not model_path:
            raise ValueError(
                "enable_chat=True requires a model_path pointing at a "
                "local GGUF model."
            )

        self.total_steps = total_steps
        self.log_maxlen = log_maxlen
        self.refresh_hz = refresh_hz
        self.enable_chat = enable_chat
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.system_prompt = system_prompt
        self.log_file = log_file
        self.report_file = report_file

        self.logs: Deque[str] = deque(maxlen=log_maxlen)
        self.chat_history: Deque[str] = deque(maxlen=50)
        self.status = "Initialising…"

        self._cmd_queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._llm: Any = None
        self._live: Any = None
        self._console: Any = None
        self._threads: list = []

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, trainer: "Any") -> None:
        # A callback can be attached to more than one training run. Start
        # each run with fresh worker state and a fresh view of its logs.
        self._stop_event.clear()
        self._threads = []
        self.logs.clear()
        self.chat_history.clear()
        while True:
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break

        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.live import Live
            from rich.panel import Panel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RichDashboardCallback requires the 'rich' package. "
                "Install with: pip install rich"
            ) from exc

        self._rich = {"Layout": Layout, "Live": Live, "Panel": Panel}
        self._console = Console()

        if self.enable_chat:
            self._load_llm()

        self.status = f"Training on backend={trainer.backend.name}..."
        if self.enable_chat:
            self.status += " Type a question below; enter 'quit' to close."
        self._layout = Layout()
        self._layout.split_column(
            Layout(name="upper_body", ratio=1),
            Layout(name="status_area", size=3),
        )
        self._layout["upper_body"].split_row(
            Layout(name="logs", ratio=2),
            Layout(name="chat", ratio=1),
        )

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self.refresh_hz,
            screen=False,
            transient=False,
        )
        self._live.__enter__()

        if self.enable_chat:
            chat_thread = threading.Thread(target=self._chat_worker, daemon=True)
            input_thread = threading.Thread(target=self._input_worker, daemon=True)
            chat_thread.start()
            input_thread.start()
            self._threads.extend((chat_thread, input_thread))

        t_render = threading.Thread(target=self._render_loop, daemon=True)
        t_render.start()
        self._threads.append(t_render)

    def on_epoch_end(
        self,
        trainer: "Any",
        step: int,
        loss: float,
        breakdown: Dict[str, float],
    ) -> None:
        if self.total_steps:
            elapsed = trainer.history.wall_time[-1] if trainer.history.wall_time else 0.0
            frac = step / max(self.total_steps, 1)
            line = f"step {step:>7d}/{self.total_steps} ({frac:5.1%})  loss={loss:.4e}  {elapsed:6.1f}s"
        else:
            line = f"step {step:>7d}  loss={loss:.4e}"
        terms = "  ".join(f"{k}={v:.3e}" for k, v in breakdown.items())
        if terms:
            line += f"  |  {terms}"
        with self._lock:
            self.logs.append(line)

    def on_lbfgs_begin(self, trainer: "Any") -> None:
        with self._lock:
            self.logs.append("── switching to L-BFGS fine-tuning phase ──")

    def on_train_end(self, trainer: "Any") -> None:
        best = min(trainer.history.total_loss) if trainer.history.total_loss else float("nan")
        with self._lock:
            self.logs.append(f"── training complete  |  best_loss={best:.4e} ──")
        if self.enable_chat:
            self.status = "Training complete."

        self._shutdown()

        if self.log_file:
            self._export_logs()
        if self.report_file:
            self._export_report(trainer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_llm(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "enable_chat=True requires the 'llama-cpp-python' package. "
                "Install with: pip install llama-cpp-python"
            ) from exc

        self.status = "Loading chat model…"
        self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

    def _strip_markup(self, text: str) -> str:
        return re.sub(r"\[.*?\]", "", text)

    def _render(self) -> Any:
        Panel = self._rich["Panel"]
        with self._lock:
            log_text = "\n".join(self.logs)
            chat_text = "\n".join(self.chat_history)
        self._layout["logs"].update(Panel(log_text, title="Training Logs"))
        self._layout["chat"].update(
            Panel(
                chat_text or ("(waiting for your first message)" if self.enable_chat else "(chat disabled)"),
                title="Agent Chat",
            )
        )
        self._layout["status_area"].update(Panel(self.status, title="Status"))
        return self._layout

    def _render_loop(self) -> None:
        interval = 1.0 / max(self.refresh_hz, 1)
        while not self._stop_event.is_set():
            self._live.update(self._render())
            time.sleep(interval)

    def _input_worker(self) -> None:
        """Read chat prompts without blocking the training thread."""
        while not self._stop_event.is_set():
            try:
                command = self._console.input("[bold cyan]You[/]> ").strip()
            except (EOFError, KeyboardInterrupt, OSError):
                self._stop_event.set()
                return

            if not command:
                continue
            if command.lower() in ("quit", "exit"):
                self._cmd_queue.put(command)
                return
            self._cmd_queue.put(command)

    def _chat_worker(self) -> None:
        """Runs a blocking stdin-read loop + llama.cpp inference off the
        main/training thread, so chatting never stalls the trainer."""
        while not self._stop_event.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd.lower() in ("quit", "exit"):
                self._stop_event.set()
                break

            with self._lock:
                recent_chat = list(self.chat_history)[-12:]
                self.chat_history.append(f"You: {cmd}")
                self.chat_history.append("Agent: ")
            reply_idx = len(self.chat_history) - 1

            with self._lock:
                training_status = self.status
                recent_logs = "\n".join(list(self.logs)[-20:])

            prompt = (
                f"<|system|>\n{self.system_prompt}\n\n"
                f"Current training status:\n{training_status}\n\n"
                "Latest live training results (newest last):\n"
                f"{recent_logs or '(No training results logged yet.)'}\n\n"
                "Recent conversation:\n"
                f"{chr(10).join(recent_chat) or '(No earlier messages.)'}"
                f"<|end|>\n"
                f"<|user|>\n{cmd}<|end|>\n"
                f"<|assistant|>\n"
            )
            stream = self._llm(
                prompt, max_tokens=512, stop=["<|end|>"], stream=True, temperature=0.7
            )
            for chunk in stream:
                if self._stop_event.is_set():
                    break
                token = chunk["choices"][0]["text"]
                with self._lock:
                    self.chat_history[reply_idx] += token

    def send_chat(self, message: str) -> None:
        """Queue a chat message from outside the training loop, e.g. from
        a separate input thread you control."""
        if not self.enable_chat:
            raise RuntimeError("Chat is disabled (enable_chat=False).")
        self._cmd_queue.put(message)

    def _shutdown(self) -> None:
        """Stop workers and restore the terminal after a dashboard run."""
        self._stop_event.set()
        current = threading.current_thread()
        for thread in self._threads:
            if thread is not current:
                thread.join(timeout=2.0)

        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

        self._threads = []

    def _export_logs(self) -> None:
        # Explicit UTF-8: the default `open()` encoding is the platform
        # locale's (e.g. cp1252 on Windows), which can't represent the
        # box-drawing / em-dash characters these log messages use
        # ("── switching to L-BFGS ──" etc.), raising UnicodeEncodeError
        # partway through export. UTF-8 can represent all of it and is a
        # safe, portable choice for a log file regardless of platform.
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== PhysAI Session Log ===\n")
            for log in self.logs:
                f.write(self._strip_markup(log) + "\n")
            if self.enable_chat:
                f.write("\n--- Conversation History ---\n")
                for chat in self.chat_history:
                    f.write(self._strip_markup(chat) + "\n")

    def _export_report(self, trainer: Any) -> None:
        """Write a polished PDF summary and loss-history plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.patches import FancyBboxPatch
        except ImportError as exc:  # pragma: no cover - matplotlib is a core dependency
            raise ImportError(
                "The dashboard PDF report requires matplotlib. "
                "Install it with: pip install matplotlib"
            ) from exc

        history = trainer.history
        steps = np.asarray(getattr(history, "steps", []), dtype=float)
        total = np.asarray(getattr(history, "total_loss", []), dtype=float)
        terms = getattr(history, "terms", {}) or {}
        val_loss = np.asarray(getattr(history, "val_loss", []), dtype=float)
        lr = np.asarray(getattr(history, "lr", []), dtype=float)
        wall_time = getattr(history, "wall_time", []) or []
        target = getattr(getattr(getattr(trainer, "config", None), "problem", None), "target_loss", None)

        finite_total = total[np.isfinite(total)]
        initial_loss = float(total[0]) if total.size else float("nan")
        final_loss = float(total[-1]) if total.size else float("nan")
        best_loss = float(np.min(finite_total)) if finite_total.size else float("nan")
        finite_indices = np.flatnonzero(np.isfinite(total))
        best_index = (
            int(finite_indices[np.argmin(total[finite_indices])])
            if finite_indices.size else None
        )
        best_step = int(steps[best_index]) if best_index is not None and best_index < steps.size else None
        runtime = float(wall_time[-1]) if wall_time else 0.0
        reached_target = (
            target is not None
            and np.isfinite(final_loss)
            and final_loss <= float(target)
        )
        if target is None:
            convergence = "No target configured"
        elif reached_target:
            convergence = "Target reached"
        else:
            convergence = "Target not reached"

        config = getattr(trainer, "config", None)
        problem = getattr(config, "problem", None)
        model_config = getattr(config, "model", None)
        pde_name = getattr(problem, "pde_name", "unspecified")
        backend_name = getattr(getattr(trainer, "backend", None), "name", "unspecified")
        model_name = getattr(model_config, "arch", "unspecified")
        timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

        def _fmt(value: float) -> str:
            return f"{value:.4e}" if np.isfinite(value) else "n/a"

        fig = plt.figure(figsize=(11.7, 8.3), facecolor="#f3f6fb")
        fig.text(0.07, 0.945, "PhysAI", fontsize=12, color="#168c91", weight="bold")
        fig.text(0.07, 0.895, "Training report", fontsize=25, color="#14243a", weight="bold")
        fig.text(
            0.07, 0.855,
            f"{pde_name}   ·   {backend_name}   ·   {model_name}   ·   {timestamp}",
            fontsize=9, color="#65758b",
        )

        card_values = [
            ("TRAINING STEPS", str(len(total))),
            ("INITIAL → BEST LOSS", f"{_fmt(initial_loss)}  →  {_fmt(best_loss)}"),
            ("FINAL LOSS / RUNTIME", f"{_fmt(final_loss)}  /  {runtime:.1f}s"),
            ("CONVERGENCE", convergence),
        ]
        card_lefts = [0.07, 0.30, 0.53, 0.76]
        card_width = 0.21
        for left, (label, value) in zip(card_lefts, card_values):
            card = fig.add_axes([left, 0.735, card_width, 0.085])
            card.set_facecolor("white")
            card.set_xticks([])
            card.set_yticks([])
            for spine in card.spines.values():
                spine.set_visible(False)
            card.add_patch(FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.018,rounding_size=0.06",
                transform=card.transAxes,
                facecolor="white", edgecolor="#dce4ee", linewidth=0.8,
                clip_on=False,
            ))
            card.text(0.08, 0.68, label, transform=card.transAxes,
                      fontsize=7, color="#718096", weight="bold", zorder=2)
            card.text(0.08, 0.25, value, transform=card.transAxes,
                      fontsize=10, color="#14243a", weight="bold", zorder=2)
            card.set_xlim(0, 1)
            card.set_ylim(0, 1)

        ax = fig.add_axes([0.09, 0.16, 0.72, 0.51], facecolor="white")
        ax.set_title("Loss history", loc="left", fontsize=14, color="#14243a", weight="bold", pad=14)
        ax.set_xlabel("Training step", color="#65758b")
        ax.set_ylabel("Loss", color="#65758b")
        ax.grid(True, which="both", color="#dfe6ef", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        plotted_values = []

        if steps.size and total.size:
            x_total = steps[:total.size]
            valid = np.isfinite(x_total) & np.isfinite(total[:x_total.size])
            y_total = total[:x_total.size][valid]
            x_total = x_total[valid]
            if y_total.size:
                ax.plot(x_total, y_total, color="#168c91", linewidth=2.7,
                        label="Total loss", zorder=5)
                ax.fill_between(x_total, y_total, color="#168c91", alpha=0.08)
                plotted_values.extend(y_total.tolist())

        palette = plt.get_cmap("tab20")
        for index, (name, values) in enumerate(terms.items()):
            values = np.asarray(values, dtype=float)
            count = min(steps.size, values.size)
            if count == 0:
                continue
            x_term, y_term = steps[:count], values[:count]
            valid = np.isfinite(x_term) & np.isfinite(y_term)
            if not np.any(valid):
                continue
            ax.plot(x_term[valid], y_term[valid], color=palette(index % 20),
                    linewidth=1.35, alpha=0.9, label=str(name))
            plotted_values.extend(y_term[valid].tolist())

        if val_loss.size and steps.size:
            log_every = max(int(getattr(trainer, "log_every", 1)), 1)
            val_steps = np.asarray([int(step) for step in steps if int(step) % log_every == 0])
            val_steps = val_steps[:val_loss.size]
            count = min(val_steps.size, val_loss.size)
            valid = np.isfinite(val_loss[:count])
            if count and np.any(valid):
                ax.plot(val_steps[:count][valid], val_loss[:count][valid],
                        color="#e27a45", linewidth=1.8, linestyle="--",
                        marker="o", markersize=3.5, label="Validation loss")
                plotted_values.extend(val_loss[:count][valid].tolist())

        if target is not None and np.isfinite(float(target)):
            ax.axhline(float(target), color="#9a6bb5", linewidth=1.2,
                       linestyle=(0, (4, 3)), label=f"Target ({float(target):.2e})")
            plotted_values.append(float(target))

        if plotted_values:
            if min(plotted_values) > 0:
                ax.set_yscale("log")
            elif min(plotted_values) < 0 < max(plotted_values):
                ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1.0),
                      frameon=False, fontsize=8, labelcolor="#34445a")
        else:
            ax.text(0.5, 0.5, "No loss history was recorded for this run.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#65758b", fontsize=12)

        ax.tick_params(colors="#65758b", labelsize=8)
        target_text = f"Configured target loss: {float(target):.4e}" if target is not None else "No target loss was configured."
        convergence_detail = (
            f"Best loss {_fmt(best_loss)} at step {best_step}; {target_text}."
            if best_step is not None else target_text
        )
        fig.text(0.09, 0.105, convergence_detail, fontsize=9, color="#4f6075")
        fig.text(0.07, 0.055, "PhysAI · generated at the end of the training run",
                 fontsize=8, color="#8a98aa")
        fig.text(0.93, 0.055, "1", fontsize=8, color="#8a98aa", ha="right")

        report_path = Path(self.report_file).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(report_path, metadata={
            "Title": "PhysAI Training Report",
            "Author": "PhysAI",
            "Subject": "Training losses and convergence summary",
        }) as pdf:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
