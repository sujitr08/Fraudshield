"""Start the FraudShield API and Streamlit dashboard.

For a quiet Windows launch, use Start_FraudShield.vbs. Logs are written to
logs/ and Stop_FraudShield.bat stops the background services.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

API_HEALTH_URL = "http://127.0.0.1:8000/health"
APP_HEALTH_URL = "http://127.0.0.1:8501/_stcore/health"
APP_URL = "http://localhost:8501"


def endpoint_ready(url: str, *, require_model: bool = False) -> bool:
    try:
        with urlopen(url, timeout=1.5) as response:
            if response.status != 200:
                return False
            if require_model:
                payload = json.loads(response.read().decode("utf-8"))
                return bool(payload.get("model_loaded"))
            return True
    except (OSError, URLError, ValueError):
        return False


def show_error(message: str) -> None:
    """Show a Windows message box when running under pythonw.exe."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "FraudShield", 0x10)
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def main() -> int:
    root = Path(__file__).resolve().parent
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "launcher.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )
    log = logging.getLogger("fraudshield.launcher")
    pid_path = root / "runtime_pids.json"
    api_log_path = log_dir / "realtime_api.log"
    app_log_path = log_dir / "streamlit.log"

    required = ("streamlit", "plotly", "pandas", "numpy", "sklearn", "imblearn", "joblib", "fastapi", "uvicorn", "requests")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        message = (
            "FraudShield is missing required Python packages: " + ", ".join(missing)
            + "\n\nOpen a terminal in the project folder and run:\n"
            + f'"{sys.executable}" -m pip install -r "{root / "requirements.txt"}"'
        )
        log.error(message)
        show_error(message)
        return 1

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    api_process: subprocess.Popen[bytes] | None = None
    streamlit_process: subprocess.Popen[bytes] | None = None
    api_log_file = None
    app_log_file = None
    owns_api = False

    try:
        # Reuse the API if it is already healthy; otherwise start it silently.
        if endpoint_ready(API_HEALTH_URL, require_model=True):
            log.info("A healthy API is already running; reusing it.")
        else:
            api_log_file = api_log_path.open("a", encoding="utf-8")
            api_command = [sys.executable, "-m", "uvicorn", "realtime_api:app", "--host", "127.0.0.1", "--port", "8000"]
            log.info("Starting API service.")
            api_process = subprocess.Popen(
                api_command,
                cwd=root,
                stdout=api_log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=(os.name != "nt"),
            )
            owns_api = True
            for _ in range(40):
                if endpoint_ready(API_HEALTH_URL, require_model=True):
                    log.info("API service is ready (PID %s).", api_process.pid)
                    break
                if api_process.poll() is not None:
                    break
                time.sleep(0.5)
            if not endpoint_ready(API_HEALTH_URL, require_model=True):
                log.error("API did not become healthy. See %s", api_log_path)

        # Avoid starting a duplicate dashboard if port 8501 is already serving Streamlit.
        if endpoint_ready(APP_HEALTH_URL):
            log.info("A Streamlit dashboard is already running; opening it.")
            webbrowser.open(APP_URL)
            return 0

        app_log_file = app_log_path.open("a", encoding="utf-8")
        streamlit_command = [
            sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
            "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", "8501",
        ]
        log.info("Starting Streamlit dashboard.")
        streamlit_process = subprocess.Popen(
            streamlit_command,
            cwd=root,
            stdout=app_log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
        pid_path.write_text(
            json.dumps({"api_pid": api_process.pid if owns_api and api_process else 0,
                        "streamlit_pid": streamlit_process.pid}, indent=2),
            encoding="utf-8",
        )

        for _ in range(60):
            if endpoint_ready(APP_HEALTH_URL):
                log.info("Streamlit is ready (PID %s). Opening %s", streamlit_process.pid, APP_URL)
                webbrowser.open(APP_URL)
                break
            if streamlit_process.poll() is not None:
                log.error("Streamlit exited during startup. See %s", app_log_path)
                show_error(f"FraudShield could not start the dashboard.\n\nCheck this log file:\n{app_log_path}")
                return 1
            time.sleep(0.5)
        else:
            log.error("Streamlit did not become ready within 30 seconds. See %s", app_log_path)
            show_error(f"FraudShield is taking too long to start.\n\nCheck this log file:\n{app_log_path}")
            return 1

        # Keep this hidden supervisor alive while the dashboard is in use.
        while streamlit_process.poll() is None:
            time.sleep(1)
        log.info("Streamlit dashboard stopped.")
        return streamlit_process.returncode or 0

    except Exception:
        log.exception("Unexpected launcher failure.")
        show_error(f"FraudShield encountered a startup error.\n\nCheck this log file:\n{log_path}")
        return 1
    finally:
        if streamlit_process is not None and streamlit_process.poll() is None:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()
        if owns_api and api_process is not None and api_process.poll() is None:
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
        for handle in (api_log_file, app_log_file):
            if handle is not None:
                handle.close()
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove PID file %s", pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
