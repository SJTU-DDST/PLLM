from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .api import create_app
from .config import PLLMConfig
from .controller import PLLMController
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLLM foreground-aware vLLM daemon")
    parser.add_argument("--config", type=Path, help="Path to config.toml")
    parser.add_argument("--host", help="Override API bind host")
    parser.add_argument("--port", type=int, help="Override API port")
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate policy without calling vLLM"
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = PLLMConfig.load(args.config)
    if args.host:
        config.api_host = args.host
    if args.port:
        config.api_port = args.port
    if args.dry_run:
        config.dry_run = True
    storage = Storage()
    controller = PLLMController(config, storage)
    app = create_app(controller, storage)
    controller.start()
    try:
        app.run(
            host=config.api_host,
            port=config.api_port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
