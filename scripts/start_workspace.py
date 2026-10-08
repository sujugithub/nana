"""Start Nana and the optional bundled local runtime with SSD model storage."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--model-dir", default="/Volumes/T7/NanaModels")
    parser.add_argument("--ollama-port", type=int, default=11435)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / "data/runtime/ollama/ollama"
    address = f"http://127.0.0.1:{args.ollama_port}"
    child = None
    try:
        urllib.request.urlopen(address + "/api/version", timeout=2).close()
    except Exception:
        if not runtime.exists():
            raise SystemExit("Ollama not found. Install Ollama and configure OLLAMA_BASE_URL, or use python -m webui.server.")
        model_dir = Path(args.model_dir)
        if str(model_dir).startswith("/Volumes/") and not model_dir.parents[0].exists():
            raise SystemExit("Connect the model SSD before starting the local runtime.")
        model_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OLLAMA_HOST": address, "OLLAMA_MODELS": str(model_dir), "OLLAMA_NO_CLOUD": "1"}
        log_path = ROOT / "logs/ollama-server.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("ab") as log:
            child = subprocess.Popen([str(runtime), "serve"], env=env, stdout=log, stderr=log)
        ready = False
        for _ in range(50):
            try:
                urllib.request.urlopen(address + "/api/version", timeout=1).close()
                ready = True
                break
            except Exception:
                if child.poll() is not None:
                    break
                time.sleep(.1)
        if not ready:
            child.terminate()
            raise SystemExit("Ollama did not start. Inspect logs/ollama-server.log.")
    os.environ["OLLAMA_BASE_URL"] = address
    from webui.server import main as serve
    try:
        return serve(["--port", str(args.port)] + (["--real"] if args.real else []))
    finally:
        if child:
            child.terminate()
            child.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
