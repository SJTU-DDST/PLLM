from __future__ import annotations

import argparse
import json
import time

from flask import Flask, Response, jsonify, request, stream_with_context


app = Flask(__name__)
STATE = {"sleeping": False, "level": None, "calls": []}


@app.get("/health")
def health():
    return "", 200


@app.get("/v1/models")
def models():
    return jsonify(
        {"object": "list", "data": [{"id": "mock-nemotron", "object": "model"}]}
    )


@app.get("/is_sleeping")
def is_sleeping():
    return jsonify({"is_sleeping": STATE["sleeping"]})


@app.post("/sleep")
def sleep():
    level = int(request.args.get("level", 1))
    mode = request.args.get("mode", "keep")
    STATE["calls"].append({"method": "sleep", "level": level, "mode": mode})
    STATE["sleeping"] = True
    STATE["level"] = level
    return jsonify({"status": "ok", "level": level})


@app.post("/wake_up")
def wake_up():
    tags = request.args.getlist("tags")
    STATE["calls"].append({"method": "wake_up", "tags": tags})
    if not tags or "kv_cache" in tags or "scheduling" in tags:
        STATE["sleeping"] = False
    return jsonify({"status": "ok", "tags": tags})


@app.post("/collective_rpc")
def collective_rpc():
    payload = request.get_json(silent=True) or {}
    STATE["calls"].append({"method": "collective_rpc", "payload": payload})
    return jsonify({"status": "ok"})


@app.get("/calls")
def calls():
    return jsonify(STATE)


@app.post("/reset")
def reset():
    STATE.update({"sleeping": False, "level": None, "calls": []})
    return jsonify(STATE)


@app.post("/v1/chat/completions")
def chat():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("messages") or [{"content": ""}])[-1].get("content", "")
    answer = f"Mock response: {prompt}"
    if payload.get("stream"):
        def generate():
            for token_id, word in enumerate(answer.split(), start=1):
                while STATE["sleeping"]:
                    time.sleep(0.01)
                event = {"choices": [{"delta": {"content": word + " "}}]}
                if payload.get("return_token_ids"):
                    event["choices"][0]["token_ids"] = [token_id]
                yield f"data: {json.dumps(event)}\n\n"
                time.sleep(0.02)
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), content_type="text/event-stream")
    deadline = time.monotonic() + 10.0
    while STATE["sleeping"] and time.monotonic() < deadline:
        time.sleep(0.01)
    if STATE["sleeping"]:
        return jsonify({"error": {"message": "model is sleeping"}}), 503
    return jsonify(
        {
            "id": "mock-chat",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": answer}}],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
