from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer --no-sandbox"
)

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the PLLM Vue dashboard")
    parser.add_argument("--url", default="http://127.0.0.1:17860")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    view = QWebEngineView()
    view.resize(args.width, args.height)

    def loaded(ok: bool) -> None:
        if not ok:
            raise SystemExit(f"failed to load {args.url}")

        def capture() -> None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not view.grab().save(str(args.output)):
                raise SystemExit(f"failed to write {args.output}")
            app.quit()

        QTimer.singleShot(2500, capture)

    view.loadFinished.connect(loaded)
    view.load(QUrl(args.url))
    view.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
