from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QPoint, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)


STATE_LABELS = {
    "active": "AI 可用",
    "elastic_resident": "Decode 弹性驻留",
    "yielding": "请求已冻结",
    "quiescing": "正在让出 GPU",
    "hibernated": "深度休眠",
    "restoring": "正在恢复",
    "hot_sleep": "热休眠",
    "cold_sleep": "深度休眠",
    "waking": "正在恢复",
    "error": "需要处理",
}

STATE_COLORS = {
    "active": "#37d67a",
    "elastic_resident": "#35c6d0",
    "yielding": "#f1b84b",
    "quiescing": "#f1b84b",
    "hibernated": "#65a9ff",
    "restoring": "#35c6d0",
    "hot_sleep": "#65a9ff",
    "cold_sleep": "#8795a8",
    "waking": "#35c6d0",
    "error": "#ff6b6b",
}

MODE_OPTIONS = [
    ("自动", "auto"),
    ("AI 优先", "ai_priority"),
    ("前台优先", "foreground_priority"),
    ("保持休眠", "keep_sleeping"),
]

RESOURCE_SERIES = (
    ("vllm", "vLLM", "#42d392"),
    ("blender", "Blender", "#ffb454"),
    ("other", "其他进程", "#61a8ff"),
)


class ApiClient(QWidget):
    status_received = Signal(dict)
    events_received = Signal(list)
    replays_received = Signal(list)
    request_failed = Signal(str)

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.network = QNetworkAccessManager(self)

    def get_status(self) -> None:
        self._request("GET", "/api/v1/status", self.status_received)

    def get_events(self) -> None:
        self._request("GET", "/api/v1/events?limit=20", self.events_received, "events")

    def get_replays(self) -> None:
        self._request(
            "GET", "/api/v1/replays?limit=20", self.replays_received, "replays"
        )

    def action(self, action: str, level: int | None = None) -> None:
        payload: dict[str, Any] = {"action": action}
        if level is not None:
            payload["level"] = level
        self._request("POST", "/api/v1/actions", self.status_received, body=payload)

    def set_mode(self, mode: str) -> None:
        self._request("PUT", "/api/v1/policy", None, body={"mode": mode})

    def replay(self, replay_id: str) -> None:
        self._request("POST", f"/api/v1/replays/{replay_id}", None, body={})

    def _request(
        self,
        method: str,
        path: str,
        signal,
        envelope: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        request = QNetworkRequest(QUrl(f"{self.base_url}{path}"))
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        data = QByteArray(json.dumps(body or {}).encode("utf-8"))
        if method == "GET":
            reply = self.network.get(request)
        elif method == "PUT":
            reply = self.network.put(request, data)
        else:
            reply = self.network.post(request, data)
        reply.finished.connect(
            lambda: self._finish(reply, signal=signal, envelope=envelope)
        )

    def _finish(self, reply: QNetworkReply, signal, envelope: str | None) -> None:
        try:
            raw = bytes(reply.readAll()).decode("utf-8", errors="replace")
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.request_failed.emit(reply.errorString())
                return
            payload = json.loads(raw) if raw else {}
            if signal is not None:
                signal.emit(payload.get(envelope, []) if envelope else payload)
        except (ValueError, TypeError) as exc:
            self.request_failed.emit(str(exc))
        finally:
            reply.deleteLater()


class MultiSeriesSparkline(QWidget):
    def __init__(self, maximum: float = 100.0) -> None:
        super().__init__()
        self.maximum = max(1.0, float(maximum))
        self.values = {
            key: deque([0.0] * 60, maxlen=60)
            for key, _label, _color in RESOURCE_SERIES
        }
        self.setFixedHeight(42)

    def add_values(self, values: dict[str, float], maximum: float | None = None) -> None:
        if maximum is not None:
            self.maximum = max(1.0, float(maximum))
        for key in self.values:
            value = float(values.get(key, 0.0))
            self.values[key].append(max(0.0, min(self.maximum, value)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = QRectF(0, 3, self.width(), self.height() - 6)
        painter.fillRect(area, QColor("#151b20"))
        painter.setPen(QPen(QColor("#2d383e"), 1, Qt.PenStyle.DashLine))
        for ratio in (0.25, 0.5, 0.75):
            y = area.bottom() - ratio * area.height()
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        for key, _label, color in RESOURCE_SERIES:
            history = self.values[key]
            if len(history) < 2:
                continue
            path = QPainterPath()
            width_step = area.width() / (len(history) - 1)
            for index, value in enumerate(history):
                point = QPointF(
                    area.left() + index * width_step,
                    area.bottom() - (value / self.maximum) * area.height(),
                )
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.setPen(QPen(QColor(color), 2))
            painter.drawPath(path)


def process_resource_breakdown(status: dict[str, Any]) -> dict[str, dict[str, float]]:
    result = {
        key: {"compute": 0.0, "memory_gb": 0.0}
        for key, _label, _color in RESOURCE_SERIES
    }
    services = status.get("services") or []
    vllm_pids: set[int] = set()
    for service in services:
        pid = int(service.get("pid") or 0)
        if pid > 0:
            vllm_pids.add(pid)
        vllm_pids.update(
            int(item) for item in service.get("related_pids") or [] if int(item) > 0
        )

    sensor = status.get("sensor") or {}
    foreground = sensor.get("foreground") or {}
    foreground_pid = int(foreground.get("pid") or 0)
    foreground_text = " ".join(
        str(foreground.get(key) or "") for key in ("app_id", "title", "wm_class")
    ).lower()
    for process in sensor.get("processes") or []:
        pid = int(process.get("pid") or 0)
        name = str(process.get("name") or "").lower()
        if pid in vllm_pids or "vllm" in name:
            category = "vllm"
        elif "blender" in name or (pid == foreground_pid and "blender" in foreground_text):
            category = "blender"
        else:
            category = "other"
        result[category]["compute"] += max(0.0, float(process.get("sm_util") or 0))
        result[category]["memory_gb"] += max(
            0.0, float(process.get("memory_gb") or 0)
        )

    process_compute = sum(item["compute"] for item in result.values())
    total_compute = max(0.0, float(sensor.get("gpu_util") or 0))
    result["other"]["compute"] += max(0.0, total_compute - process_compute)
    process_memory = sum(item["memory_gb"] for item in result.values())
    total_memory = max(0.0, float(sensor.get("gpu_memory_used_gb") or 0))
    result["other"]["memory_gb"] += max(0.0, total_memory - process_memory)
    for item in result.values():
        item["compute"] = min(100.0, item["compute"])
    return result


class OverlayWindow(QWidget):
    def __init__(
        self, api_base: str, tray_enabled: bool = True, auto_connect: bool = True
    ) -> None:
        super().__init__()
        self.api = ApiClient(api_base)
        self.expanded = False
        self.drag_origin: QPoint | None = None
        self._updating_mode = False
        self._replays: list[dict[str, Any]] = []
        self.setObjectName("root")
        self.setWindowTitle("PLLM")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(370)
        self._build_ui()
        self._connect()
        self._apply_style()
        self._set_expanded(False)
        self._create_tray() if tray_enabled else None
        self._position_window()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.api.get_status)
        if auto_connect:
            self.status_timer.start(500)
        self.detail_timer = QTimer(self)
        self.detail_timer.timeout.connect(self._refresh_details)
        if auto_connect:
            self.detail_timer.start(2500)
            QTimer.singleShot(0, self.api.get_status)

    def _build_ui(self) -> None:
        shell = QFrame(self)
        shell.setObjectName("shell")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(shell)
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.brand = QLabel("PLLM")
        self.brand.setObjectName("brand")
        self.state_dot = QLabel("●")
        self.state_dot.setObjectName("stateDot")
        self.state_label = QLabel("连接中")
        self.state_label.setObjectName("stateLabel")
        self.expand_button = QPushButton("展开")
        self.expand_button.setObjectName("quietButton")
        self.expand_button.setFixedWidth(54)
        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        self.close_button.setFixedSize(26, 26)
        self.close_button.setToolTip("关闭 PLLM")
        self.close_button.setAccessibleName("关闭 PLLM")
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header.addWidget(self.brand)
        header.addSpacing(8)
        header.addWidget(self.state_dot)
        header.addWidget(self.state_label)
        header.addStretch()
        header.addWidget(self.expand_button)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        self.reason_label = QLabel("等待守护进程状态")
        self.reason_label.setObjectName("reason")
        self.reason_label.setWordWrap(True)
        self.reason_label.setMinimumHeight(34)
        layout.addWidget(self.reason_label)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(16)
        metrics.setVerticalSpacing(5)
        self.gpu_label = QLabel("GPU  --")
        self.memory_label = QLabel("显存  --")
        self.power_label = QLabel("功耗  --")
        self.foreground_label = QLabel("前台  未知")
        for item in (
            self.gpu_label,
            self.memory_label,
            self.power_label,
            self.foreground_label,
        ):
            item.setObjectName("metric")
        metrics.addWidget(self.gpu_label, 0, 0)
        metrics.addWidget(self.memory_label, 0, 1)
        metrics.addWidget(self.power_label, 1, 0)
        metrics.addWidget(self.foreground_label, 1, 1)
        layout.addLayout(metrics)

        resource_panel = QFrame()
        resource_panel.setObjectName("resourcePanel")
        resource_layout = QVBoxLayout(resource_panel)
        resource_layout.setContentsMargins(10, 8, 10, 8)
        resource_layout.setSpacing(5)
        resource_layout.addWidget(_section_label("进程资源占用"))
        resource_grid = QGridLayout()
        resource_grid.setHorizontalSpacing(7)
        resource_grid.setVerticalSpacing(2)
        self.resource_labels: dict[str, QLabel] = {}
        for row, (key, name, color) in enumerate(RESOURCE_SERIES):
            swatch = QFrame()
            swatch.setFixedSize(12, 3)
            swatch.setStyleSheet(f"background: {color}; border: none;")
            name_label = QLabel(name)
            name_label.setObjectName("resourceName")
            value_label = QLabel("计算 --  ·  显存 --")
            value_label.setObjectName("resourceValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.resource_labels[key] = value_label
            resource_grid.addWidget(swatch, row, 0)
            resource_grid.addWidget(name_label, row, 1)
            resource_grid.addWidget(value_label, row, 2)
        resource_grid.setColumnStretch(2, 1)
        resource_layout.addLayout(resource_grid)

        compute_header = QHBoxLayout()
        compute_header.addWidget(_section_label("计算占用"))
        compute_header.addStretch()
        self.compute_total_label = QLabel("GPU --")
        self.compute_total_label.setObjectName("chartValue")
        compute_header.addWidget(self.compute_total_label)
        resource_layout.addLayout(compute_header)
        self.compute_chart = MultiSeriesSparkline(100.0)
        resource_layout.addWidget(self.compute_chart)

        memory_header = QHBoxLayout()
        memory_header.addWidget(_section_label("显存占用"))
        memory_header.addStretch()
        self.memory_total_label = QLabel("-- / -- GiB")
        self.memory_total_label.setObjectName("chartValue")
        memory_header.addWidget(self.memory_total_label)
        resource_layout.addLayout(memory_header)
        self.memory_chart = MultiSeriesSparkline(1.0)
        resource_layout.addWidget(self.memory_chart)
        layout.addWidget(resource_panel)

        controls = QHBoxLayout()
        self.release_button = QPushButton("立即释放")
        self.release_button.setObjectName("primaryButton")
        self.wake_button = QPushButton("唤醒")
        self.wake_button.setObjectName("secondaryButton")
        self.mode_combo = QComboBox()
        for label, value in MODE_OPTIONS:
            self.mode_combo.addItem(label, value)
        self.mode_combo.setMinimumWidth(92)
        controls.addWidget(self.release_button)
        controls.addWidget(self.wake_button)
        controls.addWidget(self.mode_combo)
        layout.addLayout(controls)

        self.details = QFrame()
        self.details.setObjectName("details")
        detail_layout = QVBoxLayout(self.details)
        detail_layout.setContentsMargins(0, 8, 0, 0)
        detail_layout.setSpacing(9)
        detail_header = QHBoxLayout()
        self.service_label = QLabel("vLLM  未发现")
        self.action_label = QLabel("最近操作  --")
        self.service_label.setObjectName("sectionValue")
        self.action_label.setObjectName("sectionValue")
        detail_header.addWidget(self.service_label)
        detail_header.addStretch()
        detail_header.addWidget(self.action_label)
        detail_layout.addLayout(detail_header)

        self.expert_label = QLabel("Decode  idle · slots -- · route --")
        self.state_island_label = QLabel("KV/Mamba 状态小岛  --")
        self.expert_label.setObjectName("sectionValue")
        self.state_island_label.setObjectName("sectionValue")
        detail_layout.addWidget(self.expert_label)
        detail_layout.addWidget(self.state_island_label)

        detail_layout.addWidget(_section_label("事件记录"))
        self.event_list = QListWidget()
        self.event_list.setFixedHeight(106)
        detail_layout.addWidget(self.event_list)

        replay_header = QHBoxLayout()
        replay_header.addWidget(_section_label("可重放请求"))
        replay_header.addStretch()
        self.replay_button = QPushButton("重新执行")
        self.replay_button.setObjectName("quietButton")
        self.replay_button.setEnabled(False)
        replay_header.addWidget(self.replay_button)
        detail_layout.addLayout(replay_header)
        self.replay_list = QListWidget()
        self.replay_list.setFixedHeight(72)
        detail_layout.addWidget(self.replay_list)
        layout.addWidget(self.details)

    def _connect(self) -> None:
        self.api.status_received.connect(self.update_status)
        self.api.events_received.connect(self.update_events)
        self.api.replays_received.connect(self.update_replays)
        self.api.request_failed.connect(self.show_offline)
        self.expand_button.clicked.connect(lambda: self._set_expanded(not self.expanded))
        self.close_button.clicked.connect(self._quit_application)
        self.release_button.clicked.connect(lambda: self.api.action("hibernate"))
        self.wake_button.clicked.connect(lambda: self.api.action("wake"))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.replay_list.currentRowChanged.connect(
            lambda row: self.replay_button.setEnabled(row >= 0)
        )
        self.replay_button.clicked.connect(self._replay_selected)

    def _create_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        menu = QMenu()
        show_action = QAction("显示 PLLM", self)
        dashboard_action = QAction("打开控制中心", self)
        release_action = QAction("立即释放 vLLM", self)
        wake_action = QAction("唤醒 vLLM", self)
        quit_action = QAction("退出界面", self)
        show_action.triggered.connect(self.showNormal)
        dashboard_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.api.base_url))
        )
        release_action.triggered.connect(lambda: self.api.action("hibernate"))
        wake_action.triggered.connect(lambda: self.api.action("wake"))
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(show_action)
        menu.addAction(dashboard_action)
        menu.addSeparator()
        menu.addAction(release_action)
        menu.addAction(wake_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda _reason: self.showNormal())
        self.tray.show()

    def update_status(self, status: dict[str, Any]) -> None:
        state = str(status.get("state", "error"))
        color = STATE_COLORS.get(state, STATE_COLORS["error"])
        self.state_dot.setStyleSheet(f"color: {color};")
        self.state_label.setText(STATE_LABELS.get(state, state))
        self.reason_label.setText(str(status.get("reason") or "系统空闲"))
        sensor = status.get("sensor") or {}
        gpu_util = float(sensor.get("gpu_util") or 0)
        self.gpu_label.setText(f"GPU  {gpu_util:.0f}%")
        used = sensor.get("gpu_memory_used_gb")
        total = sensor.get("gpu_memory_total_gb")
        self.memory_label.setText(
            f"显存  {used:.1f}/{total:.0f} GiB"
            if isinstance(used, (int, float)) and isinstance(total, (int, float))
            else f"内存  {float(sensor.get('memory_available_gb') or 0):.0f} GiB 可用"
        )
        power = sensor.get("power_watts")
        self.power_label.setText(
            f"功耗  {float(power):.0f} W" if isinstance(power, (int, float)) else "功耗  --"
        )
        foreground = sensor.get("foreground") or {}
        app = foreground.get("app_id") or foreground.get("wm_class") or "未识别"
        self.foreground_label.setText(f"前台  {str(app)[:18]}")
        resources = process_resource_breakdown(status)
        for key, _name, _color in RESOURCE_SERIES:
            values = resources[key]
            self.resource_labels[key].setText(
                f"计算 {values['compute']:.0f}%  ·  显存 {values['memory_gb']:.1f} GiB"
            )
        self.compute_total_label.setText(f"GPU {gpu_util:.0f}%")
        self.compute_chart.add_values(
            {key: resources[key]["compute"] for key, _name, _color in RESOURCE_SERIES}
        )
        memory_maximum = float(total) if isinstance(total, (int, float)) else 1.0
        self.memory_total_label.setText(
            f"{float(used):.1f} / {memory_maximum:.0f} GiB"
            if isinstance(used, (int, float)) and isinstance(total, (int, float))
            else "-- / -- GiB"
        )
        self.memory_chart.add_values(
            {
                key: resources[key]["memory_gb"]
                for key, _name, _color in RESOURCE_SERIES
            },
            maximum=memory_maximum,
        )
        services = status.get("services") or []
        controllable = sum(1 for item in services if item.get("controllable"))
        self.service_label.setText(f"vLLM  {controllable}/{len(services)} 可控制")
        duration = status.get("last_action_duration_ms")
        reclaimed = status.get("reclaimed_gb")
        if isinstance(reclaimed, (int, float)):
            self.action_label.setText(f"释放 {reclaimed:.1f} GiB")
        elif isinstance(duration, (int, float)):
            self.action_label.setText(f"操作 {duration:.0f} ms")
        else:
            self.action_label.setText("最近操作  --")
        residency = status.get("expert_residency") or {}
        data_plane = residency.get("data_plane") or {}
        decode_plan = residency.get("decode_plan") or {}
        route_trace = data_plane.get("route_trace") or {}
        state_island = data_plane.get("state_island") or {}
        phase = str(route_trace.get("phase") or "idle")
        slots_by_layer = data_plane.get("slots_by_layer") or decode_plan.get(
            "slots_by_layer"
        ) or {}
        slot_values = [int(value) for value in slots_by_layer.values()]
        if slot_values:
            slots = (
                str(min(slot_values))
                if min(slot_values) == max(slot_values)
                else f"{min(slot_values)}-{max(slot_values)}"
            )
        else:
            slots = data_plane.get(
                "slots_per_layer", decode_plan.get("slots_per_layer")
            )
        windows = (route_trace.get("next_window") or {}).get(
            "minimum_completed_windows", 0
        )
        horizon = (decode_plan.get("horizon") or {}).get("remaining_tokens", 0)
        self.expert_label.setText(
            f"Decode  {phase} · slots {slots or '--'}/512 · W {windows} · H {horizon}"
        )
        island_bytes = int(state_island.get("allocated_bytes") or 0)
        guard = state_island.get("resize_guard") or {}
        guard_text = "preserved" if guard.get("preserved") is True else "pending"
        island_text = f"{island_bytes / 1024**2:.0f} MiB" if island_bytes else "--"
        self.state_island_label.setText(
            f"KV/Mamba 状态小岛  {island_text} · {guard_text}"
        )
        self.release_button.setEnabled(
            state in {"active", "elastic_resident", "yielding"}
            and controllable > 0
        )
        self.wake_button.setEnabled(
            state in {"yielding", "hibernated", "hot_sleep", "cold_sleep", "error"}
        )
        mode = str(status.get("mode", "auto"))
        index = self.mode_combo.findData(mode)
        if index >= 0 and index != self.mode_combo.currentIndex():
            self._updating_mode = True
            self.mode_combo.setCurrentIndex(index)
            self._updating_mode = False

    def update_events(self, events: list[dict[str, Any]]) -> None:
        self.event_list.clear()
        for event in events[:8]:
            self.event_list.addItem(
                f"{event.get('event_type', '')}  {str(event.get('reason', ''))[:42]}"
            )

    def update_replays(self, replays: list[dict[str, Any]]) -> None:
        self._replays = [item for item in replays if item.get("status") != "completed"]
        self.replay_list.clear()
        for item in self._replays:
            request_data = item.get("request") or {}
            messages = request_data.get("messages") or []
            prompt = messages[-1].get("content", "") if messages else "未命名请求"
            token = item.get("paused_at_token") or item.get("generated_tokens") or 0
            self.replay_list.addItem(
                f"{item.get('status')} @ token {token}  {str(prompt)[:28]}"
            )

    def show_offline(self, message: str) -> None:
        self.state_dot.setStyleSheet(f"color: {STATE_COLORS['error']};")
        self.state_label.setText("守护进程离线")
        self.reason_label.setText(message)
        self.release_button.setEnabled(False)
        self.wake_button.setEnabled(False)

    def load_demo(self) -> None:
        self.update_status(
            {
                "state": "elastic_resident",
                "mode": "auto",
                "reason": "DEMO SCENARIO: 11 high-locality layers use 384 slots",
                "last_action_duration_ms": None,
                "reclaimed_gb": 1.8,
                "services": [
                    {
                        "controllable": True,
                        "pid": 301,
                        "related_pids": [301, 302],
                    }
                ],
                "sensor": {
                    "gpu_util": 22,
                    "gpu_memory_used_gb": 13.4,
                    "gpu_memory_total_gb": 97.9,
                    "power_watts": 126,
                    "foreground": {"pid": 401, "app_id": "Blender"},
                    "processes": [
                        {
                            "pid": 302,
                            "name": "VLLM::EngineCore",
                            "memory_gb": 8.7,
                            "sm_util": 7,
                        },
                        {
                            "pid": 401,
                            "name": "blender",
                            "memory_gb": 3.6,
                            "sm_util": 13,
                        },
                        {
                            "pid": 501,
                            "name": "other-cuda",
                            "memory_gb": 0.8,
                            "sm_util": 2,
                        },
                    ],
                },
                "expert_residency": {
                    "decode_plan": {
                        "slots_per_layer": 384,
                        "slots_by_layer": {
                            str(layer): (384 if layer < 11 else 512)
                            for layer in range(40)
                        },
                        "horizon": {"remaining_tokens": 256},
                    },
                    "data_plane": {
                        "slots_per_layer": 384,
                        "route_trace": {
                            "phase": "decode",
                            "decode_observations": 880,
                            "next_window": {"minimum_completed_windows": 2},
                        },
                        "state_island": {
                            "allocated_bytes": 441450496,
                            "resize_guard": {
                                "preserved": True,
                                "content_sampled": True,
                            },
                        },
                    },
                },
            }
        )
        self.update_events(
            [
                {
                    "event_type": "expert_dataplane",
                    "reason": "scenario: per-layer 384/512 decode",
                },
                {"event_type": "policy", "reason": "strict latency guardrail"},
            ]
        )
        self.update_replays(
            [
                {
                    "id": "demo",
                    "status": "paused",
                    "generated_tokens": 197,
                    "paused_at_token": 197,
                    "request": {"messages": [{"content": "总结当前项目进展"}]},
                }
            ]
        )
        for step in range(40):
            self.compute_chart.add_values(
                {
                    "vllm": 6.0 + (step % 7),
                    "blender": 10.0 + ((step * 3) % 13),
                    "other": 1.0 + (step % 3),
                }
            )
            self.memory_chart.add_values(
                {"vllm": 8.7, "blender": 3.2 + (step % 5) * 0.1, "other": 1.1},
                maximum=97.9,
            )

    def _mode_changed(self, _index: int) -> None:
        if not self._updating_mode:
            self.api.set_mode(str(self.mode_combo.currentData()))

    def _replay_selected(self) -> None:
        row = self.replay_list.currentRow()
        if 0 <= row < len(self._replays):
            self.api.replay(str(self._replays[row]["id"]))
            QTimer.singleShot(500, self.api.get_replays)

    def _refresh_details(self) -> None:
        if self.expanded:
            self.api.get_events()
            self.api.get_replays()

    def _quit_application(self) -> None:
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.hide()
        application = QApplication.instance()
        if application is not None:
            application.quit()

    def _set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.details.setVisible(expanded)
        self.expand_button.setText("收起" if expanded else "展开")
        self.setFixedHeight(830 if expanded else 510)

    def _position_window(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            area = screen.availableGeometry()
            self.move(area.right() - self.width() - 24, area.top() + 46)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 62:
            self.drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.drag_origin = None
        super().mouseReleaseEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#root { background: transparent; color: #edf1f3; font-size: 13px; }
            QFrame#shell { background: #20272c; border: 1px solid #3a464d; border-radius: 8px; }
            QLabel#brand { font-size: 17px; font-weight: 700; color: #ffffff; }
            QLabel#stateDot { font-size: 12px; }
            QLabel#stateLabel { color: #dce4e7; font-weight: 600; }
            QLabel#reason { color: #aebbc1; line-height: 1.2; }
            QLabel#metric { color: #d5dde0; padding: 2px 0; }
            QLabel#sectionLabel { color: #8fa0a8; font-size: 11px; font-weight: 600; }
            QLabel#sectionValue { color: #c9d3d7; font-size: 12px; }
            QLabel#resourceName { color: #c9d3d7; font-size: 12px; font-weight: 600; }
            QLabel#resourceValue { color: #dce4e7; font-size: 12px; }
            QLabel#chartValue { color: #aebbc1; font-size: 11px; }
            QFrame#resourcePanel { background: #192126; border: 1px solid #354047; border-radius: 5px; }
            QPushButton { min-height: 30px; border-radius: 5px; padding: 0 11px; font-weight: 600; }
            QPushButton#primaryButton { background: #35b978; color: #07150f; border: 1px solid #47cc8b; }
            QPushButton#primaryButton:hover { background: #45ca89; }
            QPushButton#secondaryButton { background: #35414a; color: #e8edef; border: 1px solid #50606a; }
            QPushButton#quietButton { min-height: 25px; background: transparent; color: #9faeb5; border: 1px solid #46535b; padding: 0 8px; }
            QPushButton#closeButton { min-width: 26px; min-height: 26px; max-width: 26px; max-height: 26px; background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; }
            QPushButton#closeButton:hover { background: #b94a52; border-color: #d16068; }
            QPushButton#closeButton:pressed { background: #8f343b; }
            QPushButton:disabled { background: #293137; color: #637078; border-color: #394249; }
            QComboBox { min-height: 30px; background: #151b20; color: #e3e9eb; border: 1px solid #46545d; border-radius: 5px; padding: 0 8px; }
            QComboBox QAbstractItemView { background: #20272c; color: #edf1f3; selection-background-color: #355d4a; }
            QFrame#details { border-top: 1px solid #354047; }
            QListWidget { background: #151b20; border: 1px solid #354047; border-radius: 5px; color: #bfcace; padding: 4px; outline: none; }
            QListWidget::item { min-height: 22px; }
            QListWidget::item:selected { background: #2d5946; color: #ffffff; }
            """
        )


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLLM desktop overlay")
    parser.add_argument("--api-base", default="http://127.0.0.1:17860")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--no-tray", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("PLLM")
    app.setQuitOnLastWindowClosed(args.no_tray or bool(args.screenshot))
    window = OverlayWindow(
        args.api_base,
        tray_enabled=not args.no_tray,
        auto_connect=not args.demo,
    )
    if args.expanded:
        window._set_expanded(True)
    if args.demo:
        window.status_timer.stop()
        window.detail_timer.stop()
        window.load_demo()
    window.show()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)

        def save_and_exit() -> None:
            window.grab().save(str(args.screenshot))
            app.quit()

        QTimer.singleShot(300, save_and_exit)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
