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

from typing import Optional

from config import ROUTE_LOCAL, ROUTE_REMOTE
from router import Router, RoutingDecision
from routing.artifact import RouterArtifact, load_artifact
from schemas import Task


class LearnedRouter(Router):
    def __init__(
        self,
        artifact_path: Optional[str] = None,
        artifact: Optional[RouterArtifact] = None,
    ):
        if artifact is None:
            if not artifact_path:
                raise ValueError("LearnedRouter needs artifact_path or artifact")
            artifact = load_artifact(artifact_path)
        self.artifact = artifact
        # Router.__init__ sets up post_check state; the heuristic estimator
        # stays available for the per-signal debug breakdown in logs.
        super().__init__(threshold=artifact.threshold)

    def decide(self, task: Task) -> RoutingDecision:
        p_local = float(self.artifact.predict_p_local([task.prompt])[0])
        target = ROUTE_LOCAL if p_local >= self.threshold else ROUTE_REMOTE
        verdict = ">=" if target == ROUTE_LOCAL else "<"
        return RoutingDecision(
            target=target,
            confidence=round(p_local, 4),
            signals={"p_local": round(p_local, 4)},
            reason=(
                f"learned p_local {p_local:.3f} {verdict} threshold "
                f"{self.threshold:.3f} ({self.artifact.model_name})"
            ),
            router_kind="learned",
            artifact_version=self.artifact.version,
        )
