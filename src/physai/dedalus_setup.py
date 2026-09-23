"""
physai/dedalus_setup.py

First-run detection/consent gate for the optional Dedalus spectral-solver
backend.

Dedalus (https://dedalus-project.org) is not a normal pip-installable
dependency — it needs a Conda environment (and typically MPI/FFTW built
against it), so it can't just live in `install_requires`. PhysAI ships a
polyglot Windows/.cmd + Linux/macOS bash installer at
`physai/installer/installer.cmd` that automates the Conda setup.

This module decides *when it's okay to even mention that*, and never runs
anything on the user's system without their explicit, per-run consent:

  * On `import physai`, `notify_dedalus_status()` is called. It's a plain
    print — no prompts, no subprocess calls, no side effects on disk beyond
    the same `~/.physai/config.json` used by `chat_setup.py`.
  * In a non-interactive context (CI, containers, headless servers, piped
    stdin, known bot/automation env vars) it either says nothing or prints
    one informational line — it NEVER prompts and NEVER launches the
    installer itself, so `pip install physai` stays inert in automated
    pipelines. This is the behavior that keeps the package from looking
    like a supply-chain risk to PyPI/security scanners: no package should
    shell out or modify the system on import without a human saying yes.
  * In an interactive terminal, it prints the same notice plus a one-time
    y/n prompt offering to launch the bundled installer right now. The
    answer (yes, no, or "don't ask again") is cached, same as chat consent.
  * `physai.install_dedalus()` is also exposed for anyone who wants to
    trigger the installer explicitly and skip the notice/prompt entirely.

Usage
-----
    from physai.dedalus_setup import notify_dedalus_status, install_dedalus

    notify_dedalus_status()   # called automatically from physai/__init__.py
    install_dedalus()         # explicit, user-invoked

Silencing
---------
    physai.set_dedalus_notice_enabled(False)   # persists the choice
    # or:
    PHYSAI_NO_DEDALUS_NOTICE=1   python train.py
    PHYSAI_NO_PROMPT=1           python train.py   # (shared w/ chat_setup)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config storage — shares ~/.physai/config.json with chat_setup.py, but
# under its own keys so the two consent flows never collide.
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(os.environ.get("PHYSAI_HOME", Path.home() / ".physai"))
_CONFIG_PATH = _CONFIG_DIR / "config.json"

_NOTICE_ENV = "PHYSAI_NO_DEDALUS_NOTICE"   # any truthy value silences the notice entirely
_NO_PROMPT_ENV = "PHYSAI_NO_PROMPT"        # shared with chat_setup.py

_INSTALLER_PATH = Path(__file__).parent / "installer" / "installer.cmd"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass  # non-fatal — worst case we re-notify next run


def set_dedalus_notice_enabled(enabled: bool) -> None:
    """Persist whether the import-time Dedalus notice should show at all."""
    cfg = _load_config()
    cfg["dedalus_notice_enabled"] = bool(enabled)
    _save_config(cfg)


def reset_dedalus_notice() -> None:
    """Forget the stored answer so the notice/prompt can appear again."""
    cfg = _load_config()
    cfg.pop("dedalus_notice_enabled", None)
    cfg.pop("dedalus_notice_seen_at", None)
    _save_config(cfg)


def _env_bool(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in ("1", "yes", "true", "y", "on")


# ---------------------------------------------------------------------------
# Environment classification
# ---------------------------------------------------------------------------

# Env vars that reliably indicate "not a human at an interactive terminal
# right now" — CI systems, cloud notebook/build hosts, containers, common
# bot/automation markers. Absence of all of these plus a real TTY is what
# we treat as "safe to ask".
_AUTOMATION_ENV_MARKERS = (
    "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI",
    "JENKINS_URL", "BUILDKITE", "TRAVIS", "CIRCLECI", "APPVEYOR",
    "TEAMCITY_VERSION", "TF_BUILD", "CODEBUILD_BUILD_ID",
    "DRONE", "SEMAPHORE", "BITBUCKET_BUILD_NUMBER",
    "DOCKER_CONTAINER", "KUBERNETES_SERVICE_HOST",
    "COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE", "BINDER_LAUNCH_HOST",
    "AWS_LAMBDA_FUNCTION_NAME", "AWS_EXECUTION_ENV",
    "REPL_ID",  # Replit
)


def is_noninteractive_environment() -> bool:
    """
    Best-effort detection of CI / headless / cloud / bot contexts, where we
    must never prompt (and, per this module's policy, never auto-run the
    installer either — see module docstring).
    """
    if _env_bool(_NO_PROMPT_ENV):
        return True

    if any(os.environ.get(marker) for marker in _AUTOMATION_ENV_MARKERS):
        return True

    # No real terminal attached to stdin/stdout -> can't be an interactive
    # human session (covers piped input, subprocess-launched runs, most
    # notebook kernels when stdin isn't a tty, redirected logs, etc.)
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return True
    except (AttributeError, ValueError):
        # stdin/stdout replaced with something that doesn't support isatty()
        # (common in some sandboxed/embedded runners) -> treat as non-interactive.
        return True

    return False


# ---------------------------------------------------------------------------
# Dedalus detection
# ---------------------------------------------------------------------------

def is_dedalus_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("dedalus") is not None


# ---------------------------------------------------------------------------
# The notice / consent flow (called from physai/__init__.py)
# ---------------------------------------------------------------------------

def notify_dedalus_status(*, quiet: bool = False) -> None:
    """
    Import-time entry point. Never raises, never blocks in a
    non-interactive context, never launches the installer on its own.

    Behavior:
      * Dedalus already installed              -> do nothing.
      * Notice previously silenced by the user  -> do nothing.
      * Non-interactive environment              -> at most one quiet
        informational line (skipped entirely if `quiet=True` or
        PHYSAI_NO_DEDALUS_NOTICE is set); never prompts.
      * Interactive terminal, notice not yet
        answered                                 -> print the notice and
        offer a one-time y/n prompt to launch the bundled installer.
    """
    try:
        if is_dedalus_installed():
            return

        if _env_bool(_NOTICE_ENV):
            return

        cfg = _load_config()
        if cfg.get("dedalus_notice_enabled") is False:
            return

        if is_noninteractive_environment():
            if not quiet:
                print(
                    "[PhysAI] Optional 'dedalus' backend not found (used for "
                    "spectral cross-validation). Skipping setup prompt in "
                    "this non-interactive environment. Install it later "
                    f"with the bundled installer, or set {_NOTICE_ENV}=1 "
                    "to stop seeing this message.",
                    file=sys.stderr,
                )
            return

        # Already asked before in an interactive session and the answer
        # wasn't "silence forever" -> don't re-prompt every single import,
        # just remind once per process at most (we still return after the
        # first notify() call per run via the seen-at timestamp check below
        # being process-local is unnecessary; config persists across runs
        # instead, matching chat_setup.py's cadence).
        if cfg.get("dedalus_notice_enabled") is None:
            _prompt_interactively()
    except Exception:
        # This function must never be able to crash `import physai`.
        pass


def _prompt_interactively() -> None:
    cfg = _load_config()
    cfg["dedalus_notice_seen_at"] = time.time()
    _save_config(cfg)

    print(
        "\n[PhysAI] Optional Dedalus backend not detected.\n"
        "    Dedalus adds a real spectral PDE solver, used for classical\n"
        "    cross-validation of your trained models (Trainer.cross_validate_dedalus).\n"
        "    It needs a Conda environment, so it isn't a normal pip dependency —\n"
        "    PhysAI ships a guided installer that sets one up for you.\n"
    )

    try:
        ans = input(
            "Run the Dedalus installer now? [y/n] (won't ask again either way): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[PhysAI] No input received — skipping for now.")
        return

    if ans in ("y", "yes"):
        set_dedalus_notice_enabled(False)  # don't keep asking on future imports
        install_dedalus(confirm=False)
    elif ans in ("n", "no"):
        set_dedalus_notice_enabled(False)
        print(
            "[PhysAI] Skipped. Run it anytime with "
            "`python -m physai.dedalus_setup` or `physai.install_dedalus()`.\n"
        )
    else:
        print("[PhysAI] Unrecognized answer — skipping for now.\n")


# ---------------------------------------------------------------------------
# Explicit, user-invoked installer trigger
# ---------------------------------------------------------------------------

def install_dedalus(*, confirm: bool = True) -> int:
    """
    Launch the bundled Dedalus installer script.

    This is the ONLY function in PhysAI that shells out to modify the
    system, and it only ever runs when a human calls it directly (or
    answers "yes" to the interactive prompt in `notify_dedalus_status`,
    which calls this with `confirm=False` since the question was already
    asked once).

    Parameters
    ----------
    confirm : If True (default), ask for a final y/n confirmation before
              running anything — appropriate when called directly by user
              code that hasn't already confirmed. Set False only when the
              caller has already obtained consent this call.

    Returns
    -------
    The installer subprocess's exit code, or -1 if it wasn't run (declined,
    non-interactive with confirm=True, or the installer file is missing).
    """
    if not _INSTALLER_PATH.exists():
        print(f"[PhysAI] Installer not found at {_INSTALLER_PATH}.")
        return -1

    if confirm:
        if is_noninteractive_environment():
            print(
                "[PhysAI] Refusing to run the Dedalus installer without a "
                "human to confirm it (non-interactive environment detected). "
                "Call physai.install_dedalus(confirm=False) if you really "
                "want to force it in a script/CI context."
            )
            return -1
        try:
            ans = input(
                f"About to run {_INSTALLER_PATH} (may install Conda + a "
                "'dedalus3' environment on this machine). Continue? [y/n]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[PhysAI] Cancelled.")
            return -1
        if ans not in ("y", "yes"):
            print("[PhysAI] Cancelled.")
            return -1
    print(f"[PhysAI] Launching {_INSTALLER_PATH} ...")
    # The installer is a polyglot: valid as a Windows .cmd batch file *and*
    # as a bash script (the `:<<"::CMDLITERAL"` header is a no-op heredoc in
    # bash but a harmless label in cmd.exe). Because it has no leading `#!`
    # shebang (the Windows half has to come first), we can't rely on
    # `shell=True` to pick the right interpreter on POSIX — Python's
    # shell=True uses /bin/sh there, not bash, and the installer body uses
    # bash-only syntax (arrays, [[ ]], ${VAR^}) that breaks under dash/sh.
    # So we branch explicitly instead of letting the OS guess.
    try:
        os.chmod(_INSTALLER_PATH, os.stat(_INSTALLER_PATH).st_mode | 0o111)
    except OSError:
        pass  # harmless no-op on platforms where this doesn't apply

    if os.name == "nt":
        # cmd.exe correctly parses the batch half of the polyglot.
        result = subprocess.run([str(_INSTALLER_PATH)], shell=True)
    else:
        # Force bash explicitly rather than the default /bin/sh.
        result = subprocess.run(["bash", str(_INSTALLER_PATH)])
    return result.returncode

__all__ = [
    "is_dedalus_installed",
    "is_noninteractive_environment",
    "notify_dedalus_status",
    "install_dedalus",
    "set_dedalus_notice_enabled",
    "reset_dedalus_notice",
]


if __name__ == "__main__":
    # `python -m physai.dedalus_setup` -> explicit, always-confirmed run.
    sys.exit(install_dedalus(confirm=True))