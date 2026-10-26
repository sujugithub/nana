"""Runtime pre-router backed by a trained artifact.

Drop-in replacement for router.Router: same decide()/post_check() surface,
so main.run_task does not care which router it holds. The learned score is
P(local answer acceptable) — higher = safer to run local, the same direction
as the heuristic confidence — and the artifact's stored operating threshold
replaces CONFIDENCE_THRESHOLD.

The artifact is loaded exactly once, at construction. Construction FAILS
(ArtifactError) when the artifact is missing or incompatible — the caller
(main.build_router) decides whether that is fatal (ROUTER_MODE=learned) or a
visible fallback to heuristics (ROUTER_MODE=auto).
"""
from __future__ import annotations

import sys
from typing import Optional

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from router import Router, RoutingDecision
from routing.artifact import ArtifactError, RouterArtifact, load_artifact
from routing.identity import (
    SYNTHETIC,
    UNVERIFIABLE,
    Compatibility,
    compare,
    runtime_identity,
)
from schemas import Task


class LearnedRouter(Router):
    def __init__(
        self,
        artifact_path: Optional[str] = None,
        artifact: Optional[RouterArtifact] = None,
        allow_unverified: bool = False,
    ):
        """``allow_unverified`` is for tests only: it admits synthetic or
        unverifiable artifacts outside mock mode (with a warning). Retrain-
        or revalidation-required artifacts are always rejected."""
        if artifact is None:
            if not artifact_path:
                raise ValueError("LearnedRouter needs artifact_path or artifact")
            artifact = load_artifact(artifact_path)
        self.compatibility = self._validate_model_pair(artifact, allow_unverified)
        self.artifact = artifact
        # Router.__init__ sets up post_check state; the heuristic estimator
        # stays available for the per-signal debug breakdown in logs.
        super().__init__(threshold=artifact.threshold)

    @staticmethod
    def _validate_model_pair(artifact: RouterArtifact,
                             allow_unverified: bool = False) -> Compatibility:
        """Accept only an artifact trained for exactly this deployment
        (routing/identity.py). Synthetic and unverifiable artifacts are
        admitted only in mock mode or explicit test use."""
        probe = None
        if (settings.tier_mode.strip().lower() == "local_remote"
                and settings.local_backend.strip().lower() == "ollama"
                and not settings.mock_mode):
            from ollama_client import model_identity
            probe = model_identity
        result = compare(artifact.dataset_meta, runtime_identity(probe=probe))
        for notice in result.notices:
            print(f"WARNING: learned router: {notice}", file=sys.stderr)
        if result.ok:
            return result
        if result.status in (SYNTHETIC, UNVERIFIABLE) and (
            settings.mock_mode or allow_unverified
        ):
            print(f"WARNING: learned router accepted for mock/test use only "
                  f"({result.status}): {result.detail}", file=sys.stderr)
            return result
        raise ArtifactError(
            f"router artifact rejected ({result.status}): {result.detail}. "
            "Collect outcomes and retrain for the configured deployment; do "
            "not reuse thresholds across model pairs."
        )

    def decide(self, task: Task) -> RoutingDecision:
        # ``predict_p_local`` is the artifact-schema name retained for
        # compatibility. Semantically it is P(cheap tier is acceptable) for
        # whichever model pair produced the training dataset.
        p_local = float(self.artifact.predict_p_local([task.prompt])[0])
        target = ROUTE_LOCAL if p_local >= self.threshold else ROUTE_REMOTE
        verdict = ">=" if target == ROUTE_LOCAL else "<"
        return RoutingDecision(
            target=target,
            confidence=round(p_local, 4),
            signals={
                "p_cheap_ok": round(p_local, 4),
                "p_local": round(p_local, 4),  # legacy log compatibility
            },
            reason=(
                f"learned p_cheap_ok {p_local:.3f} {verdict} threshold "
                f"{self.threshold:.3f} ({self.artifact.model_name})"
            ),
            router_kind="learned",
            artifact_version=self.artifact.version,
        )
