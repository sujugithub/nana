"""Read-only Mac readiness. Never starts services or downloads model weights."""
import argparse
import json
from pathlib import Path
import urllib.request


def check(model_dir, address):
    directory = Path(model_dir)
    result = {"model_directory": str(directory), "storage_ready": directory.is_dir(),
              "runtime_address": address, "runtime_ready": False, "models": []}
    try:
        with urllib.request.urlopen(address.rstrip("/") + "/api/version", timeout=2) as response:
            result["runtime_version"] = json.load(response).get("version")
        with urllib.request.urlopen(address.rstrip("/") + "/api/tags", timeout=2) as response:
            result["models"] = [model["name"] for model in json.load(response).get("models", [])]
        result["runtime_ready"] = True
    except Exception:
        result["error"] = "Dedicated Ollama runtime is unavailable"
    if not result["storage_ready"]:
        result["error"] = "Connect the model SSD; Nana will not relocate weights"
    result["ready"] = result["runtime_ready"] and result["storage_ready"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/Volumes/T7/NanaModels")
    parser.add_argument("--address", default="http://127.0.0.1:11435")
    args = parser.parse_args()
    result = check(args.model_dir, args.address)
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
