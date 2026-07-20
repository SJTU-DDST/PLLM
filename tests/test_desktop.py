from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pllm.desktop import OverlayWindow, process_resource_breakdown


def test_overlay_has_top_right_close_button() -> None:
    application = QApplication.instance() or QApplication([])
    window = OverlayWindow(
        "http://127.0.0.1:17860", tray_enabled=False, auto_connect=False
    )

    assert window.close_button.text() == ""
    assert window.close_button.toolTip() == "关闭 PLLM"
    assert window.close_button.accessibleName() == "关闭 PLLM"
    assert not window.close_button.icon().isNull()
    assert window.close_button.width() == 26
    assert window.close_button.height() == 26

    window.deleteLater()
    application.processEvents()


def test_process_resources_are_split_between_vllm_blender_and_other() -> None:
    status = {
        "services": [{"pid": 100, "related_pids": [100, 101]}],
        "sensor": {
            "gpu_util": 95,
            "gpu_memory_used_gb": 25.0,
            "foreground": {"pid": 200, "app_id": "blender_blender.desktop"},
            "processes": [
                {"pid": 100, "name": "vllm", "sm_util": 30, "memory_gb": 10},
                {
                    "pid": 101,
                    "name": "VLLM::EngineCore",
                    "sm_util": 10,
                    "memory_gb": 5,
                },
                {"pid": 200, "name": "blender", "sm_util": 50, "memory_gb": 4},
                {"pid": 300, "name": "other", "sm_util": 5, "memory_gb": 3},
            ],
        },
    }

    resources = process_resource_breakdown(status)

    assert resources["vllm"] == {"compute": 40.0, "memory_gb": 15.0}
    assert resources["blender"] == {"compute": 50.0, "memory_gb": 4.0}
    assert resources["other"] == {"compute": 5.0, "memory_gb": 6.0}


def test_overlay_updates_resource_labels_and_histories() -> None:
    application = QApplication.instance() or QApplication([])
    window = OverlayWindow(
        "http://127.0.0.1:17860", tray_enabled=False, auto_connect=False
    )
    status = {
        "state": "active",
        "mode": "auto",
        "services": [{"pid": 100, "related_pids": [100], "controllable": True}],
        "sensor": {
            "gpu_util": 75,
            "gpu_memory_used_gb": 30.0,
            "gpu_memory_total_gb": 96.0,
            "foreground": {"pid": 200, "app_id": "Blender"},
            "processes": [
                {"pid": 100, "name": "vllm", "sm_util": 25, "memory_gb": 20},
                {"pid": 200, "name": "blender", "sm_util": 45, "memory_gb": 8},
            ],
        },
    }

    window.update_status(status)

    assert window.resource_labels["vllm"].text() == "计算 25%  ·  显存 20.0 GiB"
    assert window.resource_labels["blender"].text() == "计算 45%  ·  显存 8.0 GiB"
    assert window.resource_labels["other"].text() == "计算 5%  ·  显存 2.0 GiB"
    assert window.compute_chart.values["vllm"][-1] == 25.0
    assert window.memory_chart.values["blender"][-1] == 8.0
    assert window.memory_chart.maximum == 96.0

    window.deleteLater()
    application.processEvents()
