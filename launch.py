"""Start the Satellite Change Map app.

On the first run this creates a private Python environment in .venv and
installs the dependencies; afterwards it starts in a few seconds. The app
opens in your web browser. Close this window to stop it.
"""

import hashlib
import os
import socket
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / "requirements.sha1"


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment():
    python = venv_python()
    if not python.exists():
        print("First run: setting up (this takes a minute or two, only once)…")
        venv.create(VENV, with_pip=True)
    digest = hashlib.sha1(REQUIREMENTS.read_bytes()).hexdigest()
    if not STAMP.exists() or STAMP.read_text() != digest:
        print("Installing dependencies…")
        subprocess.check_call([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                               "-q", "-r", str(REQUIREMENTS)])
        STAMP.write_text(digest)
    return python


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(port, proc, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is needed (found {sys.version.split()[0]}). "
                 "Get it from https://www.python.org/downloads/")
    python = ensure_environment()
    port = free_port()
    proc = subprocess.Popen(
        [str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true", "--server.port", str(port),
         "--server.address", "localhost", "--browser.gatherUsageStats", "false"],
        cwd=ROOT,
    )
    url = f"http://localhost:{port}"
    if wait_for(port, proc):
        webbrowser.open(url)
        print(f"\nSatellite Change Map is running at {url}\nClose this window to stop it.")
    try:
        sys.exit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
