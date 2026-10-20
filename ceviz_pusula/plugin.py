"""Ceviz router plugin entry point (router contract v1).

Ceviz loads this through the ``ceviz.routers`` entry point only when the operator sets
``WATCH_CEVIZ_ROUTER=pusula``. Requests and answers are plain dicts; see Ceviz's
docs/router-plugins.md for the contract.
"""
from __future__ import annotations

from typing import Any

from .engine import CevizPusula
from .pusula_types import MODE_CONTEXT_AWARE

API_VERSION = 1
# Hosts that do not state routing_timeout_ms (older Ceviz helpers) wait 1500 ms by default.
HOST_DEFAULT_TIMEOUT_MS = 1500
# Left for the rest of the turn and the hand-off back to the host, so the answer lands in time.
HOST_MARGIN_MS = 200
MIN_DECISION_BUDGET_MS = 100


def decision_budget_ms(host: dict[str, Any]) -> int:
    """The most a turn's decisions may take: the host's routing timeout minus a margin."""
    timeout = host.get("routing_timeout_ms")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        timeout = HOST_DEFAULT_TIMEOUT_MS
    return max(MIN_DECISION_BUDGET_MS, timeout - HOST_MARGIN_MS)


def _uses_decisions(engine: CevizPusula) -> bool:
    config = engine.config
    return config.enabled and (
        engine.light_tier_enabled()
        or (config.enable_correction_escalation and config.routing_mode == MODE_CONTEXT_AWARE)
    )


class PusulaRouter:
    api_version = API_VERSION

    def __init__(self, host: dict[str, Any], engine: CevizPusula | None = None) -> None:
        if host.get("api_version") != API_VERSION:
            raise RuntimeError(f"Pusula implements Ceviz router contract v{API_VERSION}")
        if engine is None:
            engine = CevizPusula(
                state_dir=host.get("state_dir") or None,
                agent=str(host.get("agent") or "main"),
                decision_budget_ms=decision_budget_ms(host),
            )
            config = engine.config
            if not (config.escalation_models or config.escalation_model):
                # The escalation ladder comes from model discovery; have it ready before the
                # first miss so that turn does not spend the host's routing budget on the CLI.
                engine.catalog.warm()
            if _uses_decisions(engine):
                # A cold load takes seconds; loading now keeps the first turn inside the budget.
                engine.decisions.warm()
        self.engine = engine

    def prepare(self) -> None:
        """Optional contract hook: a turn is on its way, its words not yet known.

        Ceviz calls it before speech recognition and does not wait. An idle decision model gets
        one token's work now, so the turn's questions find it at full speed.
        """
        if _uses_decisions(self.engine):
            self.engine.decisions.wake()

    def route(self, request: dict[str, Any]) -> dict[str, Any] | None:
        context: dict[str, Any] = {}
        if request.get("continuation"):
            context["continuation"] = request["continuation"]
        jobs = [job for job in request.get("recent_jobs") or [] if isinstance(job, dict)]
        if jobs:
            context["recent_jobs"] = jobs
            context["recent_job"] = jobs[-1]
        decision = self.engine.route(str(request.get("transcript") or ""), context=context)
        if not decision.model and not decision.thinking:
            return None
        reason = decision.reason
        if decision.escalation_signals:
            reason += ":" + "+".join(decision.escalation_signals)
        return {"model": decision.model, "thinking": decision.thinking, "reason": reason}


def create(host: dict[str, Any]) -> PusulaRouter:
    return PusulaRouter(host)
