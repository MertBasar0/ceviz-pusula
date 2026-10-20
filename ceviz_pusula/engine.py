from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard
from .decision_model import DecisionModel
from .model_catalog import ModelCatalog
from .pusula_types import (
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    ModelEntry,
    PusulaConfig,
    RouteDecision,
)

logger = logging.getLogger("ceviz.pusula.engine")

# A job that ended here did not get the user what they asked for.
MISSED_OUTCOMES = frozenset({"blocked", "unknown"})
MAX_ESCALATION_LEVEL = 2

# Decision models read instructions literally and work best in English; the state stays the
# user's own (Turkish) words.
LIGHT_TURN_INSTRUCTIONS = (
    "Ceviz is a voice assistant that can run commands, read files, check GitHub, inspect this "
    "machine, and delegate work to other agents. Is this request only small talk or a "
    "general-knowledge question that can be answered completely from memory, without any tool, "
    "command, file, web lookup, live system state, earlier conversation, or delegation?"
)


class CevizPusula:
    """Keeps turns on the agent's own model, and climbs a ladder only when the conversation shows a miss.

    Measured on 2026-09-26: mid-size models failed tool work (no delegation, write loops,
    unverified claims), while a wrong strong route only costs latency. So there is no middle tier.

    Strong turns carry no model override. OpenClaw disables the agent's configured fallback
    chain for any explicit `--model` run, so pinning the primary would turn one provider 503
    into a failed job. Only escalations and an optional light tier pin a model.

    Escalation is stateless: a local decision model reads the transcript and the recent jobs
    Ceviz already persists, so a restart neither loses nor invents a level. The same model
    decides light turns. When it cannot answer, the turn stays on the agent default; nothing
    else guesses in its place.
    """

    def __init__(
        self,
        state_dir: Path | str | None = None,
        catalog: ModelCatalog | None = None,
        agent: str = "main",
        decisions: DecisionModel | None = None,
        decision_budget_ms: int | None = None,
    ) -> None:
        self.config_guard = ConfigGuard(state_dir=state_dir)
        self.config, self.init_status = self.config_guard.load_config()
        self._decisions_injected = decisions is not None
        # The most a turn's decisions may take, set by the host; decision_timeout_ms never exceeds it.
        self.decision_budget_ms = decision_budget_ms
        self.decisions = decisions or _decisions_for(self.config, decision_budget_ms)
        self.catalog = catalog or ModelCatalog(agent, cache_path=self.config_guard.state_dir / "pusula-models.json")
        logger.info(f"[pusula] {self.describe()}")

    def refresh(self) -> None:
        """Reloads configuration."""
        self.config, self.init_status = self.config_guard.load_config()
        if not self._decisions_injected:
            self.decisions = _decisions_for(self.config, self.decision_budget_ms)
        logger.info(f"[pusula] Refreshed: {self.describe()}")

    def describe(self) -> str:
        # No discovery here: `openclaw models list` runs only when a turn actually escalates.
        light = self._light_model()
        refs = self.config.escalation_models or ([self.config.escalation_model] if self.config.escalation_model else [])
        return (
            f"mode={self.config.routing_mode} strong=agent-default(with fallbacks) "
            f"light={light.id if light else None} light_threshold={self.config.light_threshold} "
            f"escalation={refs or 'auto(balanced)'}@{self.config.escalation_thinking} "
            f"threshold={self.config.escalation_threshold} history={self.config.escalation_history} "
            f"decisions={self.decisions.describe()}"
        )

    def route(
        self,
        prompt: str,
        context: dict[str, Any] | str | None = None,
    ) -> RouteDecision:
        start = time.perf_counter()
        clean_prompt = (prompt or "").strip()
        config = self.config
        mode = config.routing_mode

        if not config.enabled or mode == MODE_DISABLED:
            return RouteDecision(model=None, group=config.default_group, reason="pusula_disabled")

        context_used = mode != MODE_SINGLE_TURN and _has_context(context)

        def decide(model: ModelEntry | None, group: str, reason: str, **extra: Any) -> RouteDecision:
            return RouteDecision(
                model=model.id if model else None,
                group=group,
                thinking=model.thinking if model else None,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                reason=reason,
                context_used=context_used,
                **extra,
            )

        def strong(reason: str, **extra: Any) -> RouteDecision:
            return decide(None, config.default_group, reason, **extra)

        # Escalation and the light check share one budget inside the host's routing timeout.
        deadline = self.decisions.start_deadline()

        if mode != MODE_SINGLE_TURN and config.enable_correction_escalation and clean_prompt:
            verdict = self.escalation_level(clean_prompt, context, deadline=deadline)
            if verdict is None:
                logger.warning("[pusula.route] decision model did not answer; staying on the agent default")
                return strong("decision_failed", fallback=True)
            level, signals = verdict
            if level:
                ladder = self.escalation_ladder()
                rung = ladder[min(level, len(ladder)) - 1] if ladder else None
                logger.info(
                    f"[pusula.route] escalation L{level} {list(signals)} in '{clean_prompt[:40]}' -> "
                    f"{rung.id + '@' + str(rung.thinking) if rung else 'agent default'}"
                )
                return decide(
                    rung, config.default_group, f"escalation_l{level}",
                    escalated=True, escalation_level=level, escalation_signals=signals,
                )

        # A follow-up inside an ongoing task needs that task's tools and history.
        if context_used:
            return strong("active_context")

        light = self._light_model()
        if light is None or not clean_prompt:
            return strong("no_light_tier")

        probability = self.decisions.is_light(
            clean_prompt, config.light_instructions or LIGHT_TURN_INSTRUCTIONS, deadline=deadline
        )
        if probability is None:
            logger.warning("[pusula.route] light-turn check did not answer; staying on the agent default")
            return strong("decision_failed", fallback=True)

        logger.info(f"[pusula.route] light-turn p={probability:.2f} (threshold {config.light_threshold})")
        if probability >= config.light_threshold:
            return decide(light, config.light_group, "light_turn", decision_calls=1, light_probability=probability)
        return strong("strong_turn", decision_calls=1, light_probability=probability)

    def escalation_level(
        self, prompt: str, context: dict[str, Any] | str | None, deadline: float | None = None
    ) -> tuple[int, tuple[str, ...]] | None:
        """Level 1 when this turn signals a miss; level 2 when the previous turn was a miss too.

        The decision model answers two questions about this turn: is it a correction, and does it re-ask a
        recent request that did not end in `done`. Either one gives level 1. Level 2 needs the
        previous job to have ended blocked/unknown/failed, or the previous turn to have been a
        correction itself (one more question, asked only then). A plain failure with no user
        signal does not escalate: a provider outage is not a model problem.

        Returns None when the decision model could not answer, so the caller can stay on the agent default.
        """
        config = self.config
        recent = _recent_jobs(context, config.escalation_window_seconds)
        deadline = self.decisions.start_deadline() if deadline is None else deadline
        started = time.perf_counter()
        answers = self.decisions.assess(prompt, recent, deadline=deadline)
        if answers is None:
            return None
        signals = [name for name in ("correction", "repeat") if answers.get(name, 0.0) >= config.escalation_threshold]
        summary = " ".join(f"{name}={answers[name]:.2f}" for name in sorted(answers))
        if not signals:
            logger.info(
                f"[pusula.route] decisions {summary} (threshold {config.escalation_threshold}) "
                f"{(time.perf_counter() - started) * 1000:.0f} ms -> no escalation"
            )
            return 0, ()
        level = 1
        previous = recent[-1] if recent else None
        if previous is not None:
            repeated = _missed(previous)
            if not repeated:
                earlier = self.decisions.is_correction(str(previous.get("transcript") or ""), recent[:-1], deadline=deadline)
                if earlier is None:
                    return None
                summary += f" previous_correction={earlier:.2f}"
                repeated = earlier >= config.escalation_threshold
            if repeated:
                level = 2
                signals.append("repeated_miss")
        logger.info(
            f"[pusula.route] decisions {summary} (threshold {config.escalation_threshold}) "
            f"{(time.perf_counter() - started) * 1000:.0f} ms -> L{level} {signals}"
        )
        return min(level, MAX_ESCALATION_LEVEL), tuple(signals)

    def escalation_ladder(self) -> list[ModelEntry]:
        """Configured rungs, else the best available "balanced" frontier model at rising thinking."""
        config = self.config
        thinking = config.escalation_thinking or ["medium", "high"]
        refs = list(config.escalation_models)
        if not refs and config.escalation_model:
            refs = [config.escalation_model]
        if not refs:
            best = self.catalog.best("balanced") or self.catalog.best("frontier")
            refs = [best.ref] if best else []
        if not refs:
            return []
        rungs = max(len(refs), len(thinking))
        return [
            ModelEntry(id=refs[min(i, len(refs) - 1)], thinking=thinking[min(i, len(thinking) - 1)])
            for i in range(min(rungs, MAX_ESCALATION_LEVEL))
        ]

    def light_tier_enabled(self) -> bool:
        return self._light_model() is not None

    def _light_model(self) -> ModelEntry | None:
        if self.config.light_group == self.config.default_group:
            return None
        group = self.config.get_group(self.config.light_group)
        return group.primary() if group else None


def _decisions_for(config: PusulaConfig, budget_ms: int | None = None) -> DecisionModel:
    timeout_ms = config.decision_timeout_ms if budget_ms is None else min(config.decision_timeout_ms, budget_ms)
    return DecisionModel(
        url=config.decision_model_url,
        model=config.decision_model,
        timeout_ms=timeout_ms,
        history=config.escalation_history,
    )


def _missed(job: dict[str, Any]) -> bool:
    return (job.get("outcome") or "") in MISSED_OUTCOMES or job.get("status") == "failed"


def _recent_jobs(context: dict[str, Any] | str | None, window_seconds: int) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    jobs = context.get("recent_jobs") or ([context["recent_job"]] if isinstance(context.get("recent_job"), dict) else [])
    now = time.time()
    fresh = [
        job for job in jobs
        if isinstance(job, dict) and now - float(job.get("created_at") or 0) <= window_seconds
    ]
    return sorted(fresh, key=lambda job: float(job.get("created_at") or 0))


def _has_context(context: dict[str, Any] | str | None) -> bool:
    if isinstance(context, str):
        return bool(context.strip())
    if isinstance(context, dict):
        return bool(
            str(context.get("continuation") or "").strip() or context.get("recent_job") or context.get("recent_jobs")
        )
    return False
