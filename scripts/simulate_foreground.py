from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


PROFILES = {
    "idle": {"pid": 0, "app_id": "org.gnome.Terminal", "title": "Terminal", "wm_class": "Gnome-terminal"},
    "game": {"pid": os.getpid(), "app_id": "steam_app_1245620", "title": "ELDEN RING", "wm_class": "steam_app_1245620"},
    "creative": {"pid": os.getpid(), "app_id": "blender.desktop", "title": "Blender Render", "wm_class": "Blender"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a PLLM foreground test signal")
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument(
        "--output", type=Path, default=Path.home() / ".cache" / "pllm" / "foreground.json"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(PROFILES[args.profile]), encoding="utf-8")
    print(f"Foreground profile '{args.profile}' active at {args.output}")
    if args.duration > 0:
        time.sleep(args.duration)
        args.output.write_text(json.dumps(PROFILES["idle"]), encoding="utf-8")
        print("Foreground profile returned to idle")


if __name__ == "__main__":
    main()

