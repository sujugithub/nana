"""Generate an optional launchd definition. Does not install or load it."""
import argparse
from pathlib import Path
import plistlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    body = {"Label": "local.nana.workspace", "ProgramArguments": [sys.executable, str(root/"scripts/start_workspace.py")],
            "WorkingDirectory": str(root), "RunAtLoad": False, "KeepAlive": False,
            "StandardOutPath": str(root/"logs/launchd.stdout.log"),
            "StandardErrorPath": str(root/"logs/launchd.stderr.log")}
    output = Path(args.output)
    output.write_bytes(plistlib.dumps(body))
    print("Prepared", output, "(mock mode; not installed or loaded)")


if __name__ == "__main__":
    main()
