from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def pool_binary() -> Path:
    binary = Path("rdma_bridge/build/pllm-rdma-pool")
    if not binary.is_file():
        pytest.skip("pllm-rdma-pool has not been built")
    return binary


def test_pool_help_describes_both_modes(pool_binary: Path) -> None:
    result = subprocess.run(
        [str(pool_binary), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "--server" in result.stdout
    assert "--client" in result.stdout


def test_pool_rejects_invalid_pipeline_depth(pool_binary: Path) -> None:
    result = subprocess.run(
        [
            str(pool_binary),
            "--server",
            "--queue-depth",
            "33",
            "--insecure-no-auth",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "queue depth" in result.stderr


def test_pool_rejects_invalid_rdma_read_depth(pool_binary: Path) -> None:
    result = subprocess.run(
        [
            str(pool_binary),
            "--server",
            "--rd-atomic-depth",
            "0",
            "--insecure-no-auth",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "RDMA read atomic depth" in result.stderr
