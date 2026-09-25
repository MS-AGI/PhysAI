"""
physai/workbench_local.py

Local execution runtime for PhysAI Workbench (https://physai-workbench.vertex.app
and any other deployment of the physai-web frontend).

Workbench runs entirely in the browser and never sends your script or data to
any server it doesn't already trust: it talks straight to a Jupyter server on
your own machine over a token-authenticated WebSocket
(@jupyterlab/services), the same way JupyterLab itself does. This module is
the "physai-runner" side of that connection -- it starts that local server
with the right flags so the browser is allowed to reach it, and prints the
URL (with token) to paste into Workbench's "Local runtime" panel.

Install:
    pip install "physai[workbench-local]"

Run:
    physai-workbench
    physai-workbench --origin https://my-fork.vercel.app
    physai-workbench --port 8899 --open-browser

This intentionally does not import jupyter_server/ipykernel at module load
time, so `import physai` stays fast and dependency-light even when this
extra isn't installed.
"""
from __future__ import annotations

import argparse
import secrets
import socket
import sys

DEFAULT_ORIGINS = (
    "https://physai-workbench.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _missing_deps() -> list[str]:
    missing = []
    for mod in ("jupyter_server", "ipykernel"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Could not find a free port.")


def _origin_pattern(origins: list[str]) -> str:
    # jupyter_server's allow_origin_pat is a single regex checked against the
    # browser's Origin header; OR the exact origins together.
    import re
    return "|".join(re.escape(o) for o in origins)


def build_app(*, port: int, origins: list[str], token: str, open_browser: bool):
    try:
        from jupyter_server.serverapp import ServerApp
    except ImportError as exc:  # pragma: no cover - guarded by main()
        raise SystemExit(
            "physai-workbench needs the 'workbench-local' extra.\n"
            'Install it with:  pip install "physai[workbench-local]"'
        ) from exc

    app = ServerApp()
    app.initialize(argv=[
        f"--ServerApp.port={port}",
        "--ServerApp.ip=127.0.0.1",
        f"--ServerApp.token={token}",
        # Workbench's origin is not the server's own origin, so both the
        # WebSocket handshake's Origin check and the XSRF check (which
        # assumes a same-origin form submission) must be relaxed for it.
        f"--ServerApp.allow_origin_pat={_origin_pattern(origins)}",
        "--ServerApp.disable_check_xsrf=True",
        "--ServerApp.allow_remote_access=False",
        f"--ServerApp.open_browser={open_browser}",
        "--ServerApp.password=",
    ])
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="physai-workbench",
        description="Start a local Jupyter runtime that PhysAI Workbench (in your browser) can connect to.",
    )
    parser.add_argument(
        "--origin", action="append", dest="origins", metavar="URL",
        help="Workbench origin to allow (repeatable). Defaults to the official "
             "deployment plus localhost:3000 for local frontend dev.",
    )
    parser.add_argument("--port", type=int, default=8899, help="Preferred port (default: 8899).")
    parser.add_argument("--open-browser", action="store_true", help="Also open a local Jupyter tab.")
    args = parser.parse_args(argv)

    missing = _missing_deps()
    if missing:
        sys.exit(
            "Missing dependenc{}: {}\n".format("y" if len(missing) == 1 else "ies", ", ".join(missing))
            + 'Install with:  pip install "physai[workbench-local]"'
        )

    origins = args.origins or list(DEFAULT_ORIGINS)
    port = _free_port(args.port)
    token = secrets.token_urlsafe(32)

    app = build_app(port=port, origins=origins, token=token, open_browser=args.open_browser)
    url = f"http://127.0.0.1:{port}/?token={token}"

    print("PhysAI local runtime is ready.")
    print(f"Allowing requests from: {', '.join(origins)}")
    print("\nPaste this into Workbench's \"Local runtime\" field:\n")
    print(f"  {url}\n")
    print("This process must stay running while Workbench is connected. Ctrl+C to stop.")

    try:
        app.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()