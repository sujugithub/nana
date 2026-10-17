"""Small, strict JSON-schema vocabulary for Nana's tool boundary.

This deliberately rejects unsupported schema keywords rather than pretending
to validate arbitrary MCP schemas. Production registry schemas use this subset.
"""
import json
import math


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


S = {"type": "string", "maxLength": 500_000}
ID = {"type": "string", "minLength": 1, "maxLength": 300}
ANY_OBJECT = {"type": "object"}
SCHEMAS = {
    "search": obj({"query": ID}, ["query"]),
    "search_chats": obj({"query": ID}, ["query"]),
    "read_url": obj({"url": ID}, ["url"]),
    "list_items": obj({"kind": {"type": "string", "enum": ["note", "task", "document", "memory", "event", "preset", "draft", "asset", "notification", "skill"]}, "query": S}),
    "read_item": obj({"id": ID}, ["id"]),
    "save_item": obj({"kind": {"type": "string", "enum": ["note", "task", "document", "memory", "draft", "event"]}, "title": ID, "content": S, "meta": ANY_OBJECT}, ["kind", "title", "content"]),
    "read_attachment": obj({"id": ID, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 4000}}, ["id"]),
    "read_file": obj({"path": ID}, ["path"]),
    "list_files": obj({}),
    "gmail_search": obj({"query": S}),
    "gmail_read": obj({"id": ID}, ["id"]),
    "calendar_list": obj({}),
    "shell": obj({"command": {"type": "array", "items": S, "minItems": 1, "maxItems": 100}}, ["command"]),
    "mcp": obj({"server": ID, "tool": ID, "arguments": ANY_OBJECT}, ["server", "tool", "arguments"]),
    "browser": obj({"url": ID, "action": {"type": "string", "enum": ["read", "click", "fill"]}, "selector": S, "value": S}, ["url", "action"]),
    "read_skill": obj({"id": ID}, ["id"]),
    "finish": obj({"answer": S}, ["answer"]),
    "mail_send": obj({"item": ANY_OBJECT}, ["item"]),
    "calendar_publish": obj({"item": ANY_OBJECT}, ["item"]),
    "mail_labels": obj({"id": ID, "add": {"type": "array", "items": ID, "maxItems": 100}, "remove": {"type": "array", "items": ID, "maxItems": 100}}, ["id", "add", "remove"]),
    "calendar_delete": obj({"google_id": ID}, ["google_id"]),
}


def validate(value, schema, path="arguments"):
    supported = {"type", "properties", "required", "additionalProperties", "items", "enum", "minLength", "maxLength", "minItems", "maxItems", "description", "title", "$schema", "default", "minimum", "maximum"}
    if not isinstance(schema, dict) or set(schema) - supported:
        raise ValueError(f"{path}: schema vocabulary unsupported; tool cannot be executed")
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    if kind and (kind not in types or not isinstance(value, types[kind]) or kind in {"integer", "number"} and isinstance(value, bool)):
        raise ValueError(f"{path}: expected {kind}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: finite number required")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: number outside allowed range")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value outside allowed choices")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError(f"{path}: unexpected fields")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], path + "." + key)
    if isinstance(value, (str, list)):
        suffix = "Length" if isinstance(value, str) else "Items"
        if len(value) < schema.get("min" + suffix, 0) or len(value) > schema.get("max" + suffix, 500_000):
            raise ValueError(f"{path}: size outside allowed range")
    if isinstance(value, list) and "items" in schema:
        for item in value:
            validate(item, schema["items"], path + "[]")


def arguments(name, args):
    if name not in SCHEMAS:
        raise ValueError("Tool has no registered schema")
    validate(args, SCHEMAS[name])
    if len(json.dumps(args, allow_nan=False)) > 520_000:
        raise ValueError("Tool arguments exceed the limit")


def catalog(names, descriptions):
    return [{"type": "function", "function": {"name": name, "description": descriptions.get(name, "Finish with the final answer"), "parameters": SCHEMAS[name]}} for name in [*names, "finish"]]


def decision(result, enabled):
    calls = result.get("tool_calls") or []
    if result.get("protocol") == "native" and not calls:
        # Native providers finish with ordinary assistant text. This trusted
        # transport marker is set by the adapter, never parsed from model JSON.
        answer = result.get("answer")
        arguments("finish", {"answer": answer})
        if not answer.strip():
            raise ValueError("Native response contained neither a tool nor an answer")
        return {"answer": answer}
    if calls:
        if len(calls) != 1:
            raise ValueError("Request exactly one tool at a time")
        function = calls[0].get("function", {})
        args = function.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        data = {"tool": function.get("name"), "arguments": args}
        if data["tool"] == "finish":
            arguments("finish", args)
            if not args["answer"].strip():
                raise ValueError("Final answer must not be empty")
            return {"answer": args["answer"]}
    else:
        raw = result.get("answer", "").strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Return one decision object")
    if set(data) == {"answer"}:
        arguments("finish", data)
        if not data["answer"].strip():
            raise ValueError("Final answer must not be empty")
        return data
    if set(data) != {"tool", "arguments"} or data["tool"] not in enabled:
        raise ValueError("Use one enabled tool with arguments, or one final answer")
    arguments(data["tool"], data["arguments"])
    return data


def result_envelope(output=None, error=None):
    # JSON serialization is the durable result contract. Never stringify an
    # arbitrary SDK object and accidentally treat that as successful output.
    json.dumps(output, allow_nan=False)
    if error is None and isinstance(output, dict):
        if output.get("error") or output.get("isError") or output.get("exit_code", 0) != 0:
            error = str(output.get("error") or "Tool reported failure")
    return {"ok": error is None, "data": output, "error": error}
