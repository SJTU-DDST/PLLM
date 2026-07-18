from __future__ import annotations

import pytest

from scripts.calibrate_agent import restore_seconds


def test_restore_seconds_reads_segmented_level_two_result() -> None:
    row = {
        "wake_weights_seconds": 0.1,
        "reload_weights_seconds": 41.7,
        "wake_kv_seconds": 0.03,
    }

    assert restore_seconds(row) == pytest.approx(41.83)


def test_restore_seconds_prefers_direct_measurement() -> None:
    row = {
        "wake_seconds": 0.6,
        "wake_weights_seconds": 0.1,
        "reload_weights_seconds": 41.7,
    }

    assert restore_seconds(row) == 0.6
