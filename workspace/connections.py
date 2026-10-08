"""User-scoped Google OAuth and Gmail/Calendar clients; tokens never reach UI."""
import base64
from email.message import EmailMessage
import hashlib
import json
import os
import secrets
import ssl
import threading
import time
from urllib.parse import urlencode, quote
import urllib.error
import urllib.request

import certifi

from . import store

_LOCK = threading.RLock()


def secrets_read():
    try:
        return json.loads((store.home() / "connections.json").read_text())
    except FileNotFoundError:
        return {}


def secrets_write(data):
    path = store.home() / "connections.json"
    temp = path.with_suffix(".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    os.replace(temp, path)
    os.chmod(path, 0o600)


def configure(payload):
    allowed = {"google_client_id", "google_client_secret", "image_url", "image_key", "image_model"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown connection setting")
    with _LOCK:
        data = secrets_read()
        for key, value in payload.items():
            if not isinstance(value, str) or len(value) > 10_000:
                raise ValueError("Invalid connection setting")
            data[key] = value.strip()
        secrets_write(data)
    return status()


def status():
    data = secrets_read()
    return {"google_configured": bool(data.get("google_client_id")),
            "google_connected": bool(data.get("google_tokens")),
            "image_configured": bool(data.get("image_url") and data.get("image_model"))}


def http_json(url, payload=None, headers=None, form=False, method=None):
    headers = dict(headers or {})
    body = None
    if payload is not None:
        body = urlencode(payload).encode() if form else json.dumps(payload).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        # python.org macOS installations may have no configured system CA
        # bundle. Use maintained roots while retaining hostname verification.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=30, context=context) as response:
            raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError("Provider response is too large")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        # Do not forward token-bearing bodies, URLs or request headers.
        raise RuntimeError(f"Connection returned HTTP {exc.code}. Check access and service configuration.") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise RuntimeError("Connection certificate verification failed; check the CA bundle and network certificate settings") from None
        raise RuntimeError("Connection failed; check the network and service address") from None


def google_start(port):
    with _LOCK:
        data = secrets_read()
        if not data.get("google_client_id"):
            raise ValueError("Add a Google Desktop OAuth client in Connections first")
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        state = secrets.token_urlsafe(32)
        redirect = f"http://127.0.0.1:{port}/api/workspace/google/callback"
        requests = [p for p in _google_pending(data) if p.get("expires", 0) > time.time()]
        requests.append({"state": state, "verifier": verifier, "redirect": redirect, "expires": time.time() + 600})
        data.pop("google_pending", None)
        data["google_pending_requests"] = requests[-8:]
        secrets_write(data)
        params = {"client_id": data["google_client_id"], "redirect_uri": redirect,
                  "response_type": "code", "access_type": "offline", "prompt": "consent",
                  "scope": "https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/calendar.events",
                  "code_challenge": challenge, "code_challenge_method": "S256", "state": state}
    return {"url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)}


def _google_pending(data):
    """Read old saved requests too, so a restart preserves in-flight login."""
    requests = list(data.get("google_pending_requests", []))
    legacy = data.get("google_pending")
    if legacy:
        requests.append(legacy)
    return requests


def google_has_pending(state):
    with _LOCK:
        return bool(state) and any(secrets.compare_digest(state, p.get("state", ""))
                                   for p in _google_pending(secrets_read()))


def google_callback(query):
    with _LOCK:
        data = secrets_read()
        state = (query.get("state") or [""])[0]
        code = (query.get("code") or [""])[0]
        requests = _google_pending(data)
        pending = next((p for p in requests if state and secrets.compare_digest(state, p.get("state", ""))), None)
        if pending is None:
            raise ValueError("This Google sign-in link has already been used or is no longer valid. Start a new connection from Nana.")
        # Consume only the matching attempt, once, before exchanging its code.
        # An unrelated/old callback must not invalidate another tab's login.
        data.pop("google_pending", None)
        data["google_pending_requests"] = [p for p in requests if p is not pending and p.get("expires", 0) > time.time()]
        secrets_write(data)
        if pending["expires"] < time.time() or not code:
            raise ValueError("Google authorization expired, was declined or has an invalid state")
        tokens = http_json("https://oauth2.googleapis.com/token", {
            "client_id": data["google_client_id"], "client_secret": data.get("google_client_secret", ""),
            "code": code, "code_verifier": pending["verifier"], "redirect_uri": pending["redirect"],
            "grant_type": "authorization_code"}, form=True)
        tokens["expires_at"] = time.time() + tokens.get("expires_in", 3600)
        data["google_tokens"] = tokens
        secrets_write(data)


def google(path, payload=None, method=None):
    with _LOCK:
        data = secrets_read()
        tokens = data.get("google_tokens", {})
        if not tokens:
            raise ValueError("Connect your Google account in Connections first")
        if tokens.get("expires_at", 0) < time.time() + 60:
            if not tokens.get("refresh_token"):
                raise ValueError("Reconnect Google to obtain offline access")
            refreshed = http_json("https://oauth2.googleapis.com/token", {
                "client_id": data["google_client_id"], "client_secret": data.get("google_client_secret", ""),
                "refresh_token": tokens["refresh_token"], "grant_type": "refresh_token"}, form=True)
            tokens.update(refreshed)
            tokens["expires_at"] = time.time() + refreshed.get("expires_in", 3600)
            data["google_tokens"] = tokens
            secrets_write(data)
        access = tokens["access_token"]
    return http_json("https://www.googleapis.com/" + path, payload,
                     {"Authorization": "Bearer " + access}, method=method)


def disconnect():
    with _LOCK:
        data = secrets_read()
        data.pop("google_tokens", None)
        data.pop("google_pending", None)
        data.pop("google_pending_requests", None)
        secrets_write(data)
    return status()


def mail_list(query="in:inbox"):
    result = google("gmail/v1/users/me/messages?" + urlencode({"q": query, "maxResults": 20}))
    rows = []
    for item in result.get("messages", []):
        msg = google(f"gmail/v1/users/me/messages/{quote(item['id'], safe='')}?format=metadata")
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        rows.append({"id": msg["id"], "subject": headers.get("subject", "(No subject)"),
                     "from": headers.get("from", ""), "date": headers.get("date", ""),
                     "snippet": msg.get("snippet", ""), "labels": msg.get("labelIds", [])})
    return rows


def mail_read(item_id):
    result = google(f"gmail/v1/users/me/messages/{quote(item_id, safe='')}?format=full")
    texts = []
    def walk(part):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            raw = part["body"]["data"]
            texts.append(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "replace"))
        for child in part.get("parts", []):
            walk(child)
    walk(result.get("payload", {}))
    return {"id": result["id"], "content": "\n".join(texts) or result.get("snippet", "")}


def mail_send(draft_id):
    draft = store.get(draft_id)
    if draft["kind"] != "draft" or draft["meta"].get("sent") or draft["meta"].get("sending"):
        raise ValueError("Choose an unsent email draft")
    recipient = draft["meta"].get("to", "")
    if not recipient or "\n" in recipient or "\r" in recipient:
        raise ValueError("Draft needs a valid recipient")
    msg = EmailMessage()
    msg["To"] = recipient
    msg["Subject"] = draft["title"]
    msg.set_content(draft["content"])
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    store.save({"meta": {**draft["meta"], "sending": True}}, draft_id)
    # If the connection fails after submission, keep this marker: a blind
    # retry could send a duplicate. Inspect Gmail Sent before editing it.
    result = google("gmail/v1/users/me/messages/send", {"raw": raw})
    store.save({"meta": {**draft["meta"], "sending": False, "sent": True, "google_id": result["id"]}}, draft_id)
    return {"sent": True, "id": result["id"]}


def calendar_list():
    return google("calendar/v3/calendars/primary/events?" + urlencode({
        "timeMin": store.now(), "maxResults": 100, "singleEvents": "true", "orderBy": "startTime"})).get("items", [])


def calendar_publish(item_id):
    item = store.get(item_id)
    if item["kind"] != "event":
        raise ValueError("Choose a calendar event")
    meta = item["meta"]
    from datetime import datetime
    start = datetime.fromisoformat(str(meta.get("start", "")))
    end = datetime.fromisoformat(str(meta.get("end", "")))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Event requires timezone-aware start/end; end must follow start")
    event = {"summary": item["title"], "description": item["content"],
             "start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}
    path = "calendar/v3/calendars/primary/events"
    google_id = meta.get("google_id")
    if google_id:
        path += "/" + quote(google_id, safe="")
    result = google(path, event, method="PUT" if google_id else "POST")
    store.save({"meta": {**meta, "google_id": result["id"]}}, item_id)
    return result
