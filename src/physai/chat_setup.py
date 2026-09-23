"""
physai/chat_setup.py

First-run consent gate for the optional local chat model.

PhysAI's `chat` extra can pull down a local GGUF model (via
llama-cpp-python) to power an in-terminal chat/dashboard assistant.
That's a multi-GB download the user should explicitly opt into — so
before the first training run touches it, we ask once, show real
size/time numbers, and remember the answer.

Usage
-----
Call `ensure_chat_consent()` near the top of `Trainer.train()` (or any
other entry point that might want the chat model). It returns True/False
and only prompts if the user hasn't answered yet.

    from physai.chat_setup import ensure_chat_consent

    def train(self, ...):
        if ensure_chat_consent():
            _lazy_load_chat_model()
        ...

Users can also pre-answer without ever seeing a prompt:

    import physai
    physai.set_chat_enabled(True)   # or False

    # or via environment, e.g. in CI:
    #   PHYSAI_ENABLE_CHAT=yes python train.py

    # or directly, with no environment variable at all -- e.g. a
    # pipeline/config object passing its own flag straight through:
    #   ensure_chat_consent(enable=True)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model manifest — update when the bundled chat model changes.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatModelInfo:
    name: str = "Phi-3.5-mini-instruct (Q4_K_M GGUF)"
    size_bytes: int = 2_390_000_000  # ~2.39 GB
    url: str = "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    def eta_seconds(self, mbps: float) -> float:
        # mbps = megabits/sec -> bytes/sec
        return self.size_bytes / (mbps * 1e6 / 8)


CHAT_MODEL = ChatModelInfo()

_CONFIG_DIR = Path(os.environ.get("PHYSAI_HOME", Path.home() / ".physai"))
_CONFIG_PATH = _CONFIG_DIR / "config.json"
_ENV_OVERRIDE = "PHYSAI_ENABLE_CHAT"      # "yes" / "no" / "1" / "0"
_NO_PROMPT_ENV = "PHYSAI_NO_PROMPT"       # any truthy value disables prompting

# Where the actual model file lives once fetched, and how to fetch it.
_MODELS_DIR = _CONFIG_DIR / "models"
_MODEL_FILENAME = "Phi-3.5-mini-instruct-Q4_K_M.gguf"
_HF_REPO_ID = "bartowski/Phi-3.5-mini-instruct-GGUF"

# Lets the user point at a copy they already have -- e.g. one they've
# added to their own project via git-lfs -- and skip the network pull
# entirely.
_ENV_MODEL_PATH = "PHYSAI_CHAT_MODEL_PATH"


# ---------------------------------------------------------------------------
# Config read/write
# ---------------------------------------------------------------------------

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
        pass  # non-fatal — worst case we re-prompt next run


def set_chat_enabled(enabled: bool) -> None:
    """Programmatically answer the consent prompt (skips it in future runs)."""
    cfg = _load_config()
    cfg["chat_enabled"] = bool(enabled)
    cfg["chat_answered_at"] = time.time()
    _save_config(cfg)


def reset_chat_consent() -> None:
    """Forget the stored answer, so the prompt appears again next run."""
    cfg = _load_config()
    cfg.pop("chat_enabled", None)
    cfg.pop("chat_answered_at", None)
    _save_config(cfg)


# ---------------------------------------------------------------------------
# Env override parsing
# ---------------------------------------------------------------------------

def _env_bool(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in ("1", "yes", "true", "y", "on")


# ---------------------------------------------------------------------------
# Model acquisition.
#
# This is the ONLY place in PhysAI that fetches or locates the chat
# model. Consumers (e.g. physai/dashboard/live.py's chat panel) should
# call get_chat_model_path() and treat whatever comes back (a Path, or
# None) as the answer -- they should never try to download anything
# themselves.
# ---------------------------------------------------------------------------

def get_chat_model_path() -> Optional[Path]:
    """
    Resolve the local path to the chat GGUF model, downloading it if
    necessary, and return it. Never raises -- returns None if no
    usable model could be found or fetched, so callers can degrade
    gracefully (chat panel just stays off) instead of crashing.

    Resolution order:
      1. ``PHYSAI_CHAT_MODEL_PATH`` env var, if set and the file
         exists -- lets you supply your own copy (e.g. one you already
         have and have pushed to your project via git-lfs) and skip
         the network entirely.
      2. Already-downloaded file at ``~/.physai/models/<filename>``.
      3. Download from Hugging Face straight into that same location.
    """
    override = os.environ.get(_ENV_MODEL_PATH)
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
        print(
            f"[PhysAI] {_ENV_MODEL_PATH}={p} is set but no file exists "
            "there -- falling back to the default location/download."
        )

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _MODELS_DIR / _MODEL_FILENAME
    if local_path.is_file():
        return local_path

    return _download_chat_model(local_path)


def _download_chat_model(dest: Path) -> Optional[Path]:
    """
    Pull CHAT_MODEL from Hugging Face into `dest`. Returns `dest` on
    success, None on any failure -- and on failure, always tells the
    user exactly how to supply their own copy instead of retrying the
    network (e.g. via git-lfs) so a flaky/blocked connection doesn't
    become a dead end.
    """
    print(
        f"\n[PhysAI] Pulling {CHAT_MODEL.name} "
        f"(~{CHAT_MODEL.size_gb:.1f} GB, one-time) -> {dest}\n"
    )

    def _manual_fallback_hint() -> str:
        return (
            "If you already have a copy of this model (for example one "
            "you've added to your project via git-lfs), you can skip the "
            "download entirely by either:\n"
            f"    - placing the file at:  {dest}\n"
            f"    - or setting:            {_ENV_MODEL_PATH}=/path/to/your/model.gguf\n"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "[PhysAI] Could not pull the chat model: `huggingface_hub` "
            "isn't installed. Install it with `pip install physai[chat]`.\n"
            + _manual_fallback_hint()
        )
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_MODEL_FILENAME,
            local_dir=dest.parent,
        )
        result = Path(downloaded)
        if result != dest and result.is_file() and not dest.exists():
            # Older/newer huggingface_hub versions differ on exactly
            # where they place the file within local_dir; normalize.
            result.replace(dest)
            result = dest
    except Exception as exc:  # noqa: BLE001 - surface *any* pull failure, not just network ones
        print(
            f"[PhysAI] Could not pull the chat model "
            f"({exc.__class__.__name__}: {exc}).\n" + _manual_fallback_hint()
        )
        return None

    print(f"[PhysAI] Chat model ready at {result}\n")
    return result


# ---------------------------------------------------------------------------
# The consent gate
# ---------------------------------------------------------------------------

def ensure_chat_consent(
    *,
    enable: Optional[bool] = None,
    persist: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Return True if the chat model should be fetched/enabled, False
    otherwise. Prompts interactively on the first call only; the
    answer is cached in ~/.physai/config.json after that.

    In non-interactive environments (no TTY, or PHYSAI_NO_PROMPT set),
    defaults to False without blocking, unless PHYSAI_ENABLE_CHAT is
    set explicitly.

    Parameters
    ----------
    enable:
        Direct programmatic override, e.g. from a pipeline/config
        object: ``ensure_chat_consent(enable=True)``. Takes priority
        over everything else (including PHYSAI_ENABLE_CHAT) so a
        caller can turn the pipeline on/off without needing to set
        environment variables at all -- useful in notebooks, embedded
        callers, or any host process that can't easily export env
        vars for a subprocess. `None` (the default) leaves this
        override out of the decision entirely.
    persist:
        If True, an `enable=` override is written to
        ~/.physai/config.json like the env-var/interactive paths do,
        so it also applies to future runs. Defaults to False: a
        one-off argument affects only this call and won't silently
        overwrite a user's previously saved preference.
    """
    # 0. Explicit function-argument override always wins, and is the
    #    only path that doesn't require any environment variable.
    if enable is not None:
        if persist:
            set_chat_enabled(enable)
        return _finalize(enable)

    # 1. Explicit env override always wins, and also persists the answer
    #    so subsequent runs (interactive or not) stay consistent.
    env_choice = _env_bool(_ENV_OVERRIDE)
    if env_choice is not None:
        set_chat_enabled(env_choice)
        return _finalize(env_choice)

    # 2. Already answered previously (interactively or via set_chat_enabled)?
    cfg = _load_config()
    if "chat_enabled" in cfg:
        return _finalize(bool(cfg["chat_enabled"]))

    # 3. Non-interactive context: don't block, default to off, tell the
    #    user how to opt in explicitly.
    non_interactive = _env_bool(_NO_PROMPT_ENV) or not sys.stdin.isatty()
    if non_interactive:
        if not quiet:
            print(
                "[PhysAI] Chat assistant not enabled (no interactive prompt "
                f"available). To enable it, set {_ENV_OVERRIDE}=yes, or call "
                "physai.set_chat_enabled(True)."
            )
        return False

    # 4. Interactive prompt.
    return _finalize(_prompt_interactively())


def _finalize(enabled: bool) -> bool:
    """
    Common tail for every consent path that ended up True: fetch (or
    locate) the model file right now, so it's sitting in
    ~/.physai/models/ and ready *before* training/dashboard code ever
    runs -- rather than leaving discovery of "was anything actually
    downloaded?" to whatever tries to use it later.

    If consent was granted but no usable model could be found or
    pulled, chat is treated as off for this run (the stored consent
    answer itself is left untouched, so the next run just retries the
    fetch rather than re-prompting).
    """
    if not enabled:
        return False
    if get_chat_model_path() is None:
        print(
            "[PhysAI] Chat was enabled, but no usable model could be "
            "found or downloaded -- leaving chat off for this run. "
            "Resolve the issue above (or set PHYSAI_CHAT_MODEL_PATH) "
            "and re-run.\n"
        )
        return False
    return True


def _prompt_interactively() -> bool:
    info = CHAT_MODEL
    eta_fast = info.eta_seconds(mbps=100)   # decent broadband
    eta_slow = info.eta_seconds(mbps=15)    # modest connection

    def _fmt(sec: float) -> str:
        return f"~{sec / 60:.0f} min" if sec >= 60 else f"~{sec:.0f} sec"

    print(
        "\n[PhysAI] This library can download a local AI chat model to power "
        "the interactive training assistant (ask questions about your run, "
        "get PDE/debugging help, etc. — all offline once downloaded).\n"
        f"    Model:          {info.name}\n"
        f"    Download size:  {info.size_gb:.1f} GB\n"
        f"    Estimated time: {_fmt(eta_slow)} – {_fmt(eta_fast)} "
        "(depends on your connection)\n"
        "    Stored at:      ~/.physai/models/\n"
    )

    while True:
        try:
            ans = input("Download the model and enable chat now? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[PhysAI] No input received — leaving chat disabled for this run.")
            return False

        if ans in ("y", "yes"):
            set_chat_enabled(True)
            print("[PhysAI] Chat enabled -- fetching the model now.\n")
            return True
        if ans in ("n", "no"):
            set_chat_enabled(False)
            print(
                "[PhysAI] Chat disabled. You can turn it on later with "
                "physai.set_chat_enabled(True) or by re-running with "
                f"{_ENV_OVERRIDE}=yes.\n"
            )
            return False
        print("Please answer 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Manual re-run entry point.
# ---------------------------------------------------------------------------
#
# Running this file directly (`python chat_setup.py` / `python -m
# physai.chat_setup`) is treated as a deliberate signal that the user
# wants to reconsider their earlier answer -- very common when someone
# hastily typed "n" the first time a training run prompted them and
# now wants a second look. So a manual run always clears the cached
# answer and re-prompts, regardless of what's saved in
# ~/.physai/config.json. An explicit PHYSAI_ENABLE_CHAT env var, if
# set, still takes precedence, since that's an even more deliberate
# override than re-running this file.

def _main() -> None:
    reset_chat_consent()
    print(
        "[PhysAI] Re-running chat_setup.py -- your previous answer has "
        "been cleared so you can decide again.\n"
    )
    ensure_chat_consent()


if __name__ == "__main__":
    _main()


__all__ = [
    "CHAT_MODEL",
    "ChatModelInfo",
    "get_chat_model_path",
    "ensure_chat_consent",
    "set_chat_enabled",
    "reset_chat_consent",
]