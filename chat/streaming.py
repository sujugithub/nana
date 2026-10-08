"""SSE transport for existing atomic chat turns, with owner-scoped cancellation."""
import json
import re
import threading

from generation_stream import CURRENT, Stream
from workspace import store
from .service import send_message

ACTIVE = {}
LOCK = threading.RLock()


def cancel(stream_id):
    with LOCK:
        stream = ACTIVE.get((store.USER.get(), stream_id))
        if stream:
            stream.cancel.set()
    return {"stopping": bool(stream)}


def serve(handler, conversation_id, payload):
    stream_id = payload.get("stream_id", "")
    if not re.fullmatch(r"[a-f0-9-]{16,50}", stream_id):
        raise ValueError("Invalid stream ID")
    def emit(event):
        try:
            handler.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            stream.cancel.set()
            stream.check()
    stream = Stream(emit)
    with LOCK:
        key = (store.USER.get(), stream_id)
        if key in ACTIVE:
            raise ValueError("Stream is already active")
        ACTIVE[key] = stream
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    token = CURRENT.set(stream)
    try:
        emit({"type": "start", "stream_id": stream_id})
        result = send_message(conversation_id, payload.get("content", ""),
                              mock=payload.get("mock", True) is not False,
                              allow_real=handler.server.allow_real)
        emit({"type": "complete", "result": result})
    except Exception as exc:
        try:
            emit({"type": "error", "error": str(exc), "cancelled": stream.cancel.is_set()})
        except Exception:
            pass
    finally:
        if stream.cancel.is_set():
            run_id = store.run_create("cancelled_chat", {"prompt": "Interrupted chat generation"})
            store.run_save(run_id, status="cancelled", steps=stream.attempts,
                           result={"answer": "Unfinished turn discarded. Partial provider usage may be billed; absent usage is unknown, not zero."})
        CURRENT.reset(token)
        with LOCK:
            ACTIVE.pop(key, None)
