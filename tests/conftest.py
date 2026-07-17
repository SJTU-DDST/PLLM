from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from werkzeug.serving import make_server

from scripts.mock_vllm import STATE, app


@pytest.fixture()
def mock_vllm_url() -> Iterator[str]:
    STATE.update({"sleeping": False, "level": None, "calls": []})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)

