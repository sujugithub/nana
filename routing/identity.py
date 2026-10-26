"""Model-pair identity: when are recorded outcomes valid for a deployment?

A learned router predicts "will the cheap tier's answer be acceptable?" from
outcomes recorded with ONE exact configuration. This module defines that
configuration (the *pair identity*), compares a trained identity with the
running one, and fingerprints files.

The generation fields are derived from the request each client would send
(``ollama_client.build_chat_payload`` / ``remote_client.build_chat_payload``,
the same builders generation uses), for a fixed placeholder prompt. So
request-scoped overrides and code changes to message construction show up
in the identity instead of being assumed.

Field classes (docs/LEARNED_ROUTING.md, "Compatibility rules"):

RETRAIN_FIELDS     change what the models answer, so the labels change:
                   tier mode, either tier's provider/model, the cheap tier's
                   backend, digest, quantisation, chat template and model
                   parameters, and the request each tier receives (system
                   prompt, message construction, temperature, thinking mode,
                   tools). Mismatch -> "retrain_required".
REVALIDATE_FIELDS  limits that only matter for long inputs/outputs: output
                   token caps and the local context window. Mismatch ->
                   "revalidation_required". No revalidation-acceptance
                   workflow exists, so the runtime refuses the artifact.
informational      recorded, not automatically compared, but changes may
                   affect outputs and require review: the local runtime
                   version (surfaced as a notice when it differs), plus the
                   manifest ``provenance`` (hardware, OS, Python/dependency
                   versions, commit, timestamps, cost rates).

Other statuses: "unverifiable" (a required field is unknown on either side,
an older identity version, or a request-scoped override such as Chat's
messages/output limit/context profile/tools is active), and "synthetic"
(mock or toy data). Production learned routing accepts only "compatible";
"unverifiable"/"synthetic" only in explicit mock or test use.

Labels also depend on grading (grader version, quality threshold). Those
live in ``label_identity`` and are checked when outcome files are combined,
not at runtime, because the runtime does not grade.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# "1": S2 first version (no template/parameter/request hashes).
IDENTITY_VERSION = "2"
NOT_APPLICABLE = "not_applicable"
# Fixed placeholder user prompt used only to render a representative request.
PROBE_PROMPT = "<identity-probe>"

COMPATIBLE = "compatible"
RETRAIN_REQUIRED = "retrain_required"
REVALIDATION_REQUIRED = "revalidation_required"
UNVERIFIABLE = "unverifiable"
SYNTHETIC = "synthetic"

RETRAIN_FIELDS: Tuple[str, ...] = (
    "tier_mode",
    "cheap.provider",
    "cheap.backend",
    "cheap.model",
    "cheap.digest",
    "cheap.quantization",
    "cheap.template_sha256",
    "cheap.parameters_sha256",
    "cheap.think",
    "cheap.temperature",
    "cheap.system_prompt_sha256",
    "cheap.request_sha256",
    "strong.provider",
    "strong.model",
    "strong.temperature",
    "strong.system_prompt_sha256",
    "strong.request_sha256",
)
REVALIDATE_FIELDS: Tuple[str, ...] = (
    "cheap.context_tokens",
    "cheap.max_output_tokens",
    "strong.max_output_tokens",
)
# Recorded and compared only to raise a notice; never blocks on its own.
INFORMATIONAL_FIELDS: Tuple[str, ...] = ("informational.cheap_runtime_version",)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Fingerprints ─────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    """SHA-256 of the exact bytes on disk. Differs between checkouts when
    Git converts line endings; use it to verify a transferred file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    """SHA-256 of parsed JSON content in canonical form: keys sorted,
    compact separators, non-ASCII written as \\u escapes. File layout
    (indentation, CRLF vs LF between tokens) does not matter; every
    character inside strings does — nothing is normalised."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def canonical_file_sha256(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return canonical_json_sha256(json.load(fh))


# ── Building identities ──────────────────────────────────────────────────

def request_scoped_overrides() -> List[str]:
    """Per-request generation overrides currently active (Chat, agent and
    tool paths set these). Outcomes are collected without them, so a router
    is not validated for a request path that uses any of them."""
    from generation_context import LOCAL_CONTEXT, MESSAGES, OUTPUT_LIMIT, TOOLS

    active = []
    if MESSAGES.get() is not None:
        active.append("messages")
    if OUTPUT_LIMIT.get() is not None:
        active.append("output_limit")
    if LOCAL_CONTEXT.get() is not None:
        active.append("local_context")
    if TOOLS.get():
        active.append("tools")
    return active


def _request_fields(payload: Dict[str, Any], cap_key: Tuple[str, ...]) -> Dict[str, Any]:
    """Identity fields from a rendered request. The output cap and context
    window are reported separately (revalidate class) and transport-only
    keys are excluded, so request_sha256 covers everything else sent."""
    messages = payload.get("messages") or []
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    shape = copy.deepcopy(payload)
    for key in ("stream", "stream_options", "max_tokens"):
        shape.pop(key, None)
    options = shape.get("options")
    if isinstance(options, dict):
        options.pop("num_ctx", None)
        options.pop("num_predict", None)
    cap: Any = payload
    for key in cap_key:
        cap = cap.get(key) if isinstance(cap, dict) else None
    return {
        "max_output_tokens": cap,
        "system_prompt_sha256": sha256_text(system) if isinstance(system, str) else None,
        "request_sha256": canonical_json_sha256(shape),
    }


def runtime_identity(
    *,
    tier_mode: Optional[str] = None,
    local_model: Optional[str] = None,
    local_backend: Optional[str] = None,
    cheap_model: Optional[str] = None,
    strong_model: Optional[str] = None,
    probe: Optional[Callable[[str], Dict[str, Any]]] = None,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """Identity of the configured deployment, rendered from the requests the
    clients would send right now. Keyword overrides let the web UI describe
    a selected pair without mutating settings. ``probe(model)`` returns
    runtime-reported facts (ollama_client.model_identity); without it,
    digest, quantisation, template and parameters stay unknown (None)."""
    from config import settings
    from remote_client import build_chat_payload as fireworks_payload

    tier = (tier_mode or settings.tier_mode).strip().lower()
    strong_name = strong_model or settings.strong_model_name
    strong_req = fireworks_payload(strong_name, settings.strong_max_tokens, PROBE_PROMPT)
    strong = {
        "provider": "fireworks",
        "model": strong_name,
        "temperature": strong_req.get("temperature"),
        **_request_fields(strong_req, ("max_tokens",)),
    }
    informational: Dict[str, Any] = {}
    if tier == "remote_pair":
        cheap_name = cheap_model or settings.cheap_model_name
        cheap_req = fireworks_payload(cheap_name, settings.cheap_max_tokens, PROBE_PROMPT)
        cheap = {
            "provider": "fireworks",
            "backend": "fireworks",
            "model": cheap_name,
            "digest": NOT_APPLICABLE,
            "quantization": NOT_APPLICABLE,
            "template_sha256": NOT_APPLICABLE,
            "parameters_sha256": NOT_APPLICABLE,
            "think": NOT_APPLICABLE,
            "temperature": cheap_req.get("temperature"),
            "context_tokens": NOT_APPLICABLE,
            **_request_fields(cheap_req, ("max_tokens",)),
        }
    else:
        backend = (local_backend or settings.local_backend).strip().lower()
        model = local_model or settings.local_model_name
        if backend == "ollama":
            import ollama_client
            from generation_context import completion_messages

            facts: Dict[str, Any] = {}
            if probe is not None:
                try:
                    facts = probe(model) or {}
                except Exception:  # runtime down: facts stay unknown
                    facts = {}
            messages = completion_messages(PROBE_PROMPT, ollama_client.local_system_prompt(model))
            cheap_req = ollama_client.build_chat_payload(model, messages, stream=False)
            options = cheap_req.get("options") or {}
            cheap = {
                "provider": "local",
                "backend": backend,
                "model": model,
                "digest": facts.get("digest"),
                "quantization": facts.get("quantization"),
                "template_sha256": facts.get("template_sha256"),
                "parameters_sha256": facts.get("parameters_sha256"),
                "think": cheap_req.get("think"),
                "temperature": options.get("temperature"),
                "context_tokens": options.get("num_ctx"),
                **_request_fields(cheap_req, ("options", "num_predict")),
            }
            informational["cheap_runtime_version"] = facts.get("runtime_version")
        else:
            # transformers: no checkpoint revision or request builder is
            # recorded, so an identity for this backend stays unverifiable.
            cheap = {
                "provider": "local",
                "backend": backend,
                "model": model,
                "digest": None,
                "quantization": None,
                "template_sha256": None,
                "parameters_sha256": None,
                "think": NOT_APPLICABLE,
                "temperature": 0,
                "context_tokens": NOT_APPLICABLE,
                "max_output_tokens": settings.local_max_new_tokens,
                "system_prompt_sha256": sha256_text(settings.system_prompt),
                "request_sha256": None,
            }
    identity: Dict[str, Any] = {
        "identity_version": IDENTITY_VERSION,
        "tier_mode": tier,
        "cheap": cheap,
        "strong": strong,
        "informational": informational,
    }
    overrides = request_scoped_overrides()
    if overrides:
        identity["request_scoped_overrides"] = overrides
    if synthetic:
        identity["synthetic"] = True
    return identity


def _get(identity: Dict[str, Any], dotted: str) -> Any:
    value: Any = identity
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


# ── Comparing ────────────────────────────────────────────────────────────

@dataclass
class Compatibility:
    status: str
    detail: str
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    # Informational differences: never change ``status``, always surfaced.
    notices: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == COMPATIBLE

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "detail": self.detail,
                "mismatches": self.mismatches, "unknown": self.unknown,
                "notices": self.notices}


def _legacy_names(meta: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    if not meta.get("tier_mode"):
        return None
    return (str(meta["tier_mode"]).strip().lower(),
            str(meta.get("local_model", "")), str(meta.get("remote_model", "")))


def _notices(trained: Dict[str, Any], runtime: Dict[str, Any]) -> List[str]:
    out = []
    for name in INFORMATIONAL_FIELDS:
        a, b = _get(trained, name), _get(runtime, name)
        if a is not None and b is not None and a != b:
            out.append(
                f"{name.split('.', 1)[1]} differs (trained {a!r}, running {b!r}): "
                "not proof that outputs changed, nor that the old evaluation "
                "still holds — review and re-run the relevant checks")
    return out


def compare(dataset_meta: Optional[Dict[str, Any]],
            runtime: Dict[str, Any]) -> Compatibility:
    """Compare the identity a router was trained on (from its dataset meta)
    with the running deployment's identity."""
    meta = dataset_meta or {}
    trained = meta.get("pair_identity")
    if not isinstance(trained, dict):
        legacy = _legacy_names(meta)
        if legacy is None:
            return Compatibility(SYNTHETIC, (
                "artifact has no model-pair metadata (synthetic/toy training "
                "data); allowed only for mock demos and tests"))
        configured = (runtime["tier_mode"], _get(runtime, "cheap.model"),
                      _get(runtime, "strong.model"))
        if legacy != configured:
            return Compatibility(RETRAIN_REQUIRED, (
                f"router retraining required: artifact was trained for "
                f"{legacy}, configured for {configured}"),
                [{"field": "pair", "trained": list(legacy),
                  "runtime": list(configured), "class": "retrain"}])
        return Compatibility(UNVERIFIABLE, (
            "artifact records model names only (pre-identity collector); "
            "digest, quantisation and generation settings cannot be "
            "verified — collect and retrain with the current collector"))

    if trained.get("synthetic") or runtime.get("synthetic"):
        return Compatibility(SYNTHETIC, (
            "artifact was trained on mock/synthetic outcomes; allowed only for "
            "mock demos and tests"))
    if trained.get("identity_version") != IDENTITY_VERSION:
        return Compatibility(UNVERIFIABLE, (
            f"artifact identity_version {trained.get('identity_version')!r} "
            f"predates this code's {IDENTITY_VERSION!r} and lacks fields it "
            "verifies (template, parameters, rendered request); missing "
            "information is treated as unknown — collect and retrain"))
    overrides = runtime.get("request_scoped_overrides") or []
    if overrides:
        return Compatibility(UNVERIFIABLE, (
            "this request path applies request-scoped generation overrides ("
            + ", ".join(overrides) + ") that the validated batch identity "
            "does not represent; learned routing for Chat/agent paths is "
            "pending validation"), unknown=[f"request_scoped:{o}" for o in overrides])

    notices = _notices(trained, runtime)
    mismatches: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for cls, fields in (("retrain", RETRAIN_FIELDS), ("revalidate", REVALIDATE_FIELDS)):
        for name in fields:
            a, b = _get(trained, name), _get(runtime, name)
            if a is None or b is None:
                unknown.append(name)
            elif a != b:
                mismatches.append({"field": name, "trained": a, "runtime": b,
                                   "class": cls})

    retrain = [m for m in mismatches if m["class"] == "retrain"]
    if retrain:
        names = ", ".join(m["field"] for m in retrain)
        return Compatibility(RETRAIN_REQUIRED,
                             f"router retraining required: {names} differ",
                             mismatches, unknown, notices)
    if unknown:
        return Compatibility(UNVERIFIABLE, (
            "cannot verify the deployment: unknown " + ", ".join(unknown)),
            mismatches, unknown, notices)
    if mismatches:
        names = ", ".join(m["field"] for m in mismatches)
        return Compatibility(REVALIDATION_REQUIRED, (
            f"revalidation required: {names} differ. No revalidation-"
            "acceptance workflow exists; this artifact cannot be used with "
            "the changed settings. Collect outcomes under the new settings "
            "and train a replacement artifact"),
            mismatches, unknown, notices)
    detail = "artifact was trained for exactly this deployment"
    if notices:
        detail += "; review required: " + "; ".join(notices)
    return Compatibility(COMPATIBLE, detail, notices=notices)
