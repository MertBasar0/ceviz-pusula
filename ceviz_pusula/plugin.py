"""Ceviz router plugin entry point (router contract v1).

Ceviz loads this through the ``ceviz.routers`` entry point only when the operator sets
``WATCH_CEVIZ_ROUTER=pusula``. Requests and answers are plain dicts; see Ceviz's
docs/router-plugins.md for the contract.
"""
from __future__ import annotations

from typing import Any

from .engine import CevizPusula

API_VERSION = 1


class PusulaRouter:
    api_version = API_VERSION

    def __init__(self, host: dict[str, Any], engine: CevizPusula | None = None) -> None:
        if host.get("api_version") != API_VERSION:
            raise RuntimeError(f"Pusula implements Ceviz router contract v{API_VERSION}")
        if engine is None:
            engine = CevizPusula(
                state_dir=host.get("state_dir") or None,
                agent=str(host.get("agent") or "main"),
            )
            config = engine.config
            if not (config.escalation_models or config.escalation_model):
                # The escalation ladder comes from model discovery; have it ready before the
                # first miss so that turn does not spend the host's routing budget on the CLI.
                engine.catalog.warm()
        self.engine = engine

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
