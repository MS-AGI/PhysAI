"""Show the live training dashboard with simulated metrics and fake chat.

Run from the repository root after installing the dashboard extra::

    pip install "physai[dashboard]"
    python examples/live_dashboard_demo.py

The demo uses no real PDE solve, GGUF file, or llama.cpp installation. Type
a question in the terminal while the simulated training run is in progress;
the fake chat model answers from the latest dashboard log lines. Its PDF
is saved separately as ``physai_dashboard_demo.pdf``.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


class _FakeLlama:
    """Small streaming stand-in that answers from the prompt's live metrics."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        pass

    def __call__(self, prompt: str, **kwargs: Any) -> Iterator[dict]:
        results = re.findall(r"step\s+\d+/\d+[^\n]*loss=[^\s]+", prompt)
        if results:
            reply = f"The latest training result I can see is: {results[-1]}"
        else:
            reply = "I do not see a training result yet. Try asking again after the next step."

        for token in reply.split(" "):
            yield {"choices": [{"text": token + " "}]}


def _install_fake_llama_cpp() -> None:
    """Make live.py's lazy llama_cpp import resolve to the local fake model."""
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    sys.modules["llama_cpp"] = fake_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60, help="simulated training steps (default: 60)")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between steps (default: 0.3)")
    args = parser.parse_args()
    if args.steps < 1 or args.delay < 0:
        parser.error("--steps must be positive and --delay cannot be negative")

    source_dir = Path(__file__).resolve().parents[1] / "src"
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

    _install_fake_llama_cpp()
    from physai.dashboard import RichDashboardCallback

    class FakeTrainer:
        backend = SimpleNamespace(name="demo")
        history = SimpleNamespace(wall_time=[], total_loss=[])

    trainer = FakeTrainer()
    dashboard = RichDashboardCallback(
        total_steps=args.steps,
        enable_chat=True,
        model_path="fake-demo-model.gguf",
        log_file=None,
        report_file="physai_dashboard_demo.pdf",
    )

    print("Live dashboard demo: ask about the current loss while training runs.")
    print("The chat model is simulated. The training metrics are simulated too.")
    print("The demo PDF will be saved as physai_dashboard_demo.pdf.")
    dashboard.on_train_begin(trainer)

    started = time.perf_counter()
    try:
        for step in range(args.steps):
            elapsed = time.perf_counter() - started
            loss = 1.0 / (step + 1) + 0.01
            trainer.history.wall_time.append(elapsed)
            trainer.history.total_loss.append(loss)
            dashboard.on_epoch_end(
                trainer,
                step,
                loss,
                {"pde": loss * 0.8, "boundary": loss * 0.2},
            )
            if args.delay:
                time.sleep(args.delay)
    finally:
        dashboard.on_train_end(trainer)


if __name__ == "__main__":
    main()
