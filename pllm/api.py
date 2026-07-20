from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask.json.provider import DefaultJSONProvider

from .controller import PLLMController
from .storage import Storage


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class _SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs["allow_nan"] = False
        return super().dumps(_json_safe(obj), **kwargs)


def create_app(controller: PLLMController, storage: Storage) -> Flask:
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    app = Flask(__name__, static_folder=str(frontend), static_url_path="/assets")
    app.json = _SafeJSONProvider(app)

    @app.get("/")
    def dashboard():
        return send_from_directory(frontend, "index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "component": "pllm-daemon"})

    @app.get("/api/v1/status")
    def status():
        return jsonify(controller.status())

    @app.get("/api/v1/capabilities")
    def capabilities():
        return jsonify(controller.capabilities(refresh=request.args.get("refresh") == "1"))

    @app.get("/api/v1/telemetry/stream")
    def telemetry_stream():
        interval = max(0.25, min(float(request.args.get("interval", 0.5)), 5.0))

        def generate():
            while True:
                payload = app.json.dumps(controller.status(), ensure_ascii=False)
                yield f"event: status\ndata: {payload}\n\n"
                time.sleep(interval)

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/vllm")
    def vllm_services():
        if request.args.get("refresh") == "1":
            try:
                services = controller.refresh_services()
            except Exception as exc:
                return jsonify({"error": str(exc)}), 503
        else:
            services = controller.services()
        return jsonify({"services": services})

    @app.get("/api/v1/events")
    def events():
        return jsonify({"events": storage.list_events(request.args.get("limit", 100))})

    @app.get("/api/v1/replays")
    def replays():
        return jsonify(
            {"replays": storage.list_replays(request.args.get("limit", 100))}
        )

    @app.get("/api/v1/experiments")
    def experiments():
        return jsonify(
            {"experiments": storage.list_experiments(request.args.get("limit", 100))}
        )

    @app.get("/api/v1/expert-residency")
    def expert_residency():
        return jsonify(controller.expert_residency_status())

    @app.post("/api/v1/expert-residency/plan")
    def expert_residency_plan():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(controller.plan_expert_residency(payload))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.get("/api/v1/expert-dataplane")
    def expert_dataplane():
        return jsonify(controller.expert_dataplane_status())

    @app.post("/api/v1/expert-dataplane/actions")
    def expert_dataplane_actions():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(controller.expert_dataplane_action(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except OSError as exc:
            return jsonify({"error": str(exc)}), 502

    @app.put("/api/v1/policy")
    def update_policy():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify({"policy": controller.update_policy(payload)})
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/v1/policy/compile")
    def compile_policy():
        payload = request.get_json(silent=True) or {}
        try:
            result = controller.compile_policy(
                str(payload.get("text", "")), bool(payload.get("apply", False))
            )
            return jsonify(result)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/v1/actions")
    def actions():
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action", ""))
        level = payload.get("level")
        try:
            result = controller.action(action, int(level) if level is not None else None)
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except requests.RequestException as exc:
            return jsonify({"error": str(exc)}), 502

    @app.post("/v1/chat/completions")
    def chat_completions():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _openai_error("JSON request body is required", 400)

        if not controller.can_proxy():
            replay_id = storage.create_replay(payload, "queued")
            state = controller.status()["state"]
            response, status_code = _openai_error(
                f"PLLM has paused vLLM ({state}); replay_id={replay_id}", 503
            )
            response.headers["X-PLLM-Replay-ID"] = replay_id
            return response, status_code

        target = controller.proxy_target()
        if not target:
            replay_id = storage.create_replay(payload, "queued")
            response, status_code = _openai_error(
                f"No healthy vLLM backend; replay_id={replay_id}", 503
            )
            response.headers["X-PLLM-Replay-ID"] = replay_id
            return response, status_code

        replay_id = storage.create_replay(payload, "running")
        try:
            controller.prepare_inference_request(
                replay_id,
                _positive_int(payload.get("min_tokens")),
                int(payload.get("n", 1)),
            )
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            controller.mark_inference_phase("idle", request_id=replay_id)
            storage.update_replay(replay_id, "queued", error=str(exc))
            response, status_code = _openai_error(
                f"PLLM could not prepare exact expert residency; replay_id={replay_id}",
                503,
            )
            response.headers["X-PLLM-Replay-ID"] = replay_id
            return response, status_code
        if payload.get("stream"):
            return _stream_chat(target, payload, replay_id, storage, controller)
        result = _plain_chat(target, payload, replay_id, storage)
        controller.mark_inference_phase("idle", request_id=replay_id)
        return result

    @app.post("/api/v1/replays/<replay_id>")
    def replay(replay_id: str):
        entry = storage.get_replay(replay_id)
        if entry is None:
            return jsonify({"error": "Replay not found"}), 404
        if not controller.can_proxy():
            return jsonify({"error": "vLLM is still paused"}), 409
        target = controller.proxy_target()
        if not target:
            return jsonify({"error": "No healthy vLLM backend"}), 503
        payload = dict(entry["request"])
        payload["stream"] = False
        storage.update_replay(replay_id, "running")
        try:
            controller.prepare_inference_request(
                replay_id,
                _positive_int(payload.get("min_tokens")),
                int(payload.get("n", 1)),
            )
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            controller.mark_inference_phase("idle", request_id=replay_id)
            storage.update_replay(replay_id, "queued", error=str(exc))
            return jsonify({"error": str(exc)}), 503
        result = _plain_chat(target, payload, replay_id, storage)
        controller.mark_inference_phase("idle", request_id=replay_id)
        return result

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Not found"}), 404

    return app


def _plain_chat(
    target: str, payload: dict[str, Any], replay_id: str, storage: Storage
):
    try:
        response = requests.post(
            f"{target}/v1/chat/completions", json=payload, timeout=(5, 600)
        )
    except requests.RequestException as exc:
        storage.update_replay(replay_id, "aborted", error=str(exc))
        result, code = _openai_error(
            f"vLLM request interrupted; replay_id={replay_id}", 502
        )
        result.headers["X-PLLM-Replay-ID"] = replay_id
        return result, code

    content_type = response.headers.get("Content-Type", "application/json")
    if response.ok:
        response_text = _extract_assistant_text(response)
        storage.update_replay(replay_id, "completed", response_text=response_text)
    else:
        storage.update_replay(
            replay_id, "failed", error=f"vLLM returned HTTP {response.status_code}"
        )
    outgoing = Response(response.content, status=response.status_code, content_type=content_type)
    outgoing.headers["X-PLLM-Replay-ID"] = replay_id
    return outgoing


def _stream_chat(
    target: str,
    payload: dict[str, Any],
    replay_id: str,
    storage: Storage,
    controller: PLLMController,
) -> Response:
    try:
        backend_payload = dict(payload)
        client_requested_token_ids = bool(payload.get("return_token_ids"))
        backend_payload["return_token_ids"] = True
        upstream = requests.post(
            f"{target}/v1/chat/completions",
            json=backend_payload,
            stream=True,
            timeout=(5, None),
        )
        upstream.raise_for_status()
    except requests.RequestException as exc:
        controller.mark_inference_phase("idle", request_id=replay_id)
        storage.update_replay(replay_id, "aborted", error=str(exc))
        result, status = _openai_error(
            f"vLLM stream could not start; replay_id={replay_id}", 502
        )
        result.headers["X-PLLM-Replay-ID"] = replay_id
        return result, status

    def generate() -> Iterator[bytes]:
        captured: list[str] = []
        generated_tokens = 0
        decode_marked = False
        try:
            for line in upstream.iter_lines():
                if not line:
                    yield b"\n"
                    continue
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    try:
                        event = json.loads(line[6:])
                        choice = event["choices"][0]
                        delta = choice.get("delta", {})
                        token_ids = choice.get("token_ids") or []
                        content = (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                        )
                        if token_ids or content:
                            if not decode_marked:
                                controller.mark_inference_phase(
                                    "decode", request_id=replay_id
                                )
                                decode_marked = True
                        if content:
                            captured.append(str(content))
                        if token_ids:
                            generated_tokens += len(token_ids)
                            controller.record_decode_progress(
                                replay_id, len(token_ids), exact=True
                            )
                        elif content:
                            generated_tokens += 1
                            controller.record_decode_progress(
                                replay_id, 1, exact=False
                            )
                        if token_ids or content:
                            storage.update_replay_progress(
                                replay_id, generated_tokens, "".join(captured)
                            )
                        if not client_requested_token_ids:
                            choice.pop("token_ids", None)
                            event.pop("prompt_token_ids", None)
                            line = b"data: " + json.dumps(
                                event, ensure_ascii=False, separators=(",", ":")
                            ).encode("utf-8")
                    except (ValueError, KeyError, IndexError, TypeError):
                        pass
                yield line + b"\n"
            storage.update_replay(
                replay_id, "completed", response_text="".join(captured)
            )
        except requests.RequestException as exc:
            storage.update_replay(
                replay_id, "aborted", response_text="".join(captured), error=str(exc)
            )
            error = {
                "error": {
                    "message": f"PLLM interrupted the stream; replay_id={replay_id}",
                    "type": "pllm_interrupted",
                }
            }
            yield f"data: {json.dumps(error)}\n\n".encode()
        finally:
            upstream.close()
            controller.mark_inference_phase("idle", request_id=replay_id)

    response = Response(
        stream_with_context(generate()),
        status=upstream.status_code,
        content_type="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-PLLM-Replay-ID"] = replay_id
    return response


def _extract_assistant_text(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload["choices"][0]["message"].get("content", ""))
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _openai_error(message: str, status: int):
    payload = {
        "error": {
            "message": message,
            "type": "pllm_error",
            "code": status,
        }
    }
    return jsonify(payload), status
