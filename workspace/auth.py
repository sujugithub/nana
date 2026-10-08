"""Optional local accounts, scrypt password hashes and TOTP second factors."""
import base64
import hashlib
import hmac
from http.cookies import SimpleCookie
import json
import os
import re
import secrets
import struct
import threading
import time
from urllib.parse import urlencode

from . import store

_LOCK = threading.RLock()
SESSIONS = {}
FAILURES = {}


def read_users():
    try:
        return json.loads((store.ROOT / "accounts.json").read_text())
    except FileNotFoundError:
        return {}


def write_users(users):
    store.ROOT.mkdir(parents=True, exist_ok=True)
    path = store.ROOT / "accounts.json"
    fd = os.open(path.with_suffix(".tmp"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(users, fh)
    os.replace(path.with_suffix(".tmp"), path)
    os.chmod(path, 0o600)


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


def create(username, password, current=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", username) or not isinstance(password, str) or not 10 <= len(password) <= 200:
        raise ValueError("Username: 1–40 letters/numbers; password: 10–200 characters")
    with _LOCK:
        users = read_users()
        if users and (not current or not users.get(current, {}).get("admin")):
            raise PermissionError("Only the owner can add accounts")
        if username in users:
            raise ValueError("Username already exists")
        salt = secrets.token_hex(16)
        users[username] = {"salt": salt, "hash": password_hash(password, salt), "admin": not bool(users),
                           "user_id": "local" if not users else "u_" + hashlib.sha256(username.encode()).hexdigest()[:24]}
        write_users(users)
    return {"created": True}


def totp(secret, tick=None):
    tick = int(time.time() // 30) if tick is None else tick
    key = base64.b32decode(secret + "=" * (-len(secret) % 8))
    digest = hmac.new(key, struct.pack(">Q", tick), hashlib.sha1).digest()
    offset = digest[-1] & 15
    return f"{(struct.unpack('>I', digest[offset:offset+4])[0] & 0x7fffffff) % 1000000:06d}"


def valid_code(secret, code):
    return any(hmac.compare_digest(totp(secret, int(time.time() // 30) + drift), str(code)) for drift in (-1, 0, 1))


def login(payload):
    name = str(payload.get("username", ""))
    with _LOCK:
        failure = FAILURES.get(name, {"count": 0, "until": 0})
        if failure["until"] > time.time():
            raise PermissionError("Too many attempts. Try again in a minute.")
        user = read_users().get(name)
        password = payload.get("password", "")
        valid = user and isinstance(password, str) and len(password) <= 200 and hmac.compare_digest(password_hash(password, user["salt"]), user["hash"])
        if valid and user.get("totp"):
            valid = valid_code(user["totp"], payload.get("code", ""))
        if not valid:
            failure["count"] += 1
            if failure["count"] >= 5:
                failure["until"] = time.time() + 60
            FAILURES[name] = failure
            raise PermissionError("Invalid username, password or authenticator code")
        FAILURES.pop(name, None)
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = {"username": name, "user_id": user["user_id"], "expires": time.time() + 12 * 3600}
    return token


def session(headers):
    cookies = SimpleCookie()
    try:
        cookies.load(headers.get("Cookie", ""))
    except Exception:
        return None
    token = cookies["nana_session"].value if "nana_session" in cookies else ""
    with _LOCK:
        entry = SESSIONS.get(token)
        if entry and entry["expires"] > time.time():
            return entry
        SESSIONS.pop(token, None)
    return None


def setup_totp(name):
    if not name:
        raise PermissionError("Create an account and sign in first")
    secret = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
    with _LOCK:
        users = read_users()
        users[name]["pending_totp"] = secret
        write_users(users)
    return {"secret": secret, "uri": "otpauth://totp/Nana:" + name + "?" + urlencode({"secret": secret, "issuer": "Nana"})}


def confirm_totp(name, code):
    with _LOCK:
        users = read_users()
        user = users.get(name, {})
        secret = user.get("pending_totp")
        if not secret or not valid_code(secret, code):
            raise ValueError("Authenticator code did not match")
        user["totp"] = user.pop("pending_totp")
        write_users(users)
    return {"enabled": True}
