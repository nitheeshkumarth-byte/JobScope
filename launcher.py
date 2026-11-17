"""Double-click launcher for the local JobScope app.

Starts dashboard.py with the existing virtualenv's pythonw.exe (no console),
waits for the app to answer, then opens the dashboard in the browser.
Kept deliberately thin: dashboard.py still owns startup, config and errors.
"""

import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
TIMEOUT_SECONDS = 90


def app_dir() -> pathlib.Path:
    # Frozen: the .exe sits next to dashboard.py. Unfrozen: launcher.py does.
    return pathlib.Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent


def healthy(timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/api/auth/state", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_health(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if healthy():
            return True
        time.sleep(0.5)
    return False


def port_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", PORT))
        except OSError:
            return False
        return True


def fail(message: str) -> None:
    # pythonw has no console, so a raised exception would vanish silently.
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, message, "JobScope", 0x10)


def main() -> int:
    open_browser = "--no-browser" not in sys.argv[1:]

    if healthy():
        if open_browser:
            webbrowser.open(BASE + "/")
        return 0

    root = app_dir()
    pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    if not pythonw.exists():
        fail(f"JobScope could not start.\n\nMissing interpreter:\n{pythonw}\n\n"
             f"Recreate the virtualenv, or run it from a terminal to see the error.")
        return 1
    if not port_free():
        # Another launcher click may be mid-startup: give it time to answer
        # before blaming a different program for the port.
        if wait_for_health(20):
            if open_browser:
                webbrowser.open(BASE + "/")
            return 0
        fail(f"Port {PORT} is held by another program, so JobScope did not start.\n\n"
             f"Find it with:\n  Get-NetTCPConnection -LocalPort {PORT} -State Listen")
        return 1

    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "launcher.log"

    env = os.environ.copy()
    env["HOST"] = "127.0.0.1"  # never expose the app to the LAN from a double-click
    with open(log_path, "ab") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting on port {PORT} ---\n".encode())
        log.flush()
        subprocess.Popen(
            [str(pythonw), "dashboard.py", "--port", str(PORT)],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if wait_for_health(TIMEOUT_SECONDS):
        if open_browser:
            webbrowser.open(BASE + "/")
        return 0

    fail(f"JobScope did not answer on {BASE} within {TIMEOUT_SECONDS} seconds.\n\n"
         f"See the log:\n{log_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
