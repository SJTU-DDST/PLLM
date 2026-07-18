from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_client_mode_reaches_transport_instead_of_mode_rejection() -> None:
    binary = Path("rdma_bridge/build/pllm-rdma-store")
    if not binary.is_file():
        pytest.skip("pllm-rdma-store has not been built")

    result = subprocess.run(
        [
            str(binary),
            "--client",
            "127.0.0.1",
            "--port",
            "1",
            "--operation",
            "get",
            "--key",
            "probe",
            "--file",
            "/tmp/pllm-rdma-cli-probe",
            "--allocator",
            "aligned",
            "--insecure-no-auth",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "select exactly one" not in result.stderr
