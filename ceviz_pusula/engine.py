from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard
from .escalation_judge import EscalationJudge
from .jev_client import JevClient
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

# Jev reads instructions literally and works best in English; the state stays the
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
    Ceviz already persists, so a restart neither loses nor invents a level. When that model
    cannot answer, the turn stays on the agent default; nothing else guesses in its place.
    """

    def __init__(
        self,
        state_dir: Path | str | None = None,
        jev_client: JevClient | None = None,
        catalog: ModelCatalog | None = None,
        agent: str = "main",
        judge: EscalationJudge | None = None,
    ) -> None:
        self.config_guard = ConfigGuard(state_dir=state_dir)
        self.config, self.init_status = self.config_guard.load_config()
        self._jev_injected = jev_client is not None
        self.jev = jev_client or JevClient(system_one_url=self.config.decision_endpoint)
        self._judge_injected = judge is not None
        self.judge = judge or _judge_for(self.config)
        self.catalog = catalog or ModelCatalog(agent, cache_path=self.config_guard.state_dir / "pusula-models.json")
        logger.info(f"[pusula] {self.describe()}")

    def refresh(self) -> None:
        """Reloads configuration."""
        self.config, self.init_status = self.config_guard.load_config()
        if not self._jev_injected:
            self.jev = JevClient(system_one_url=self.config.decision_endpoint)
        if not self._judge_injected:
            self.judge = _judge_for(self.config)
        logger.info(f"[pusula] Refreshed: {self.describe()}")

    def describe(self) -> str:
        # No discovery here: `openclaw models list` runs only when a turn actually escalates.
        light = self._light_model()
        refs = self.config.escalation_models or ([self.config.escalation_model] if self.config.escalation_model else [])
        return (
            f"mode={self.config.routing_mode} strong=agent-default(with fallbacks) "
            f"light={light.id if light else None} light_threshold={self.config.light_threshold} "
            f"escalation={refs or 'auto(balanced)'}@{self.config.escalation_thinking} "
            f"judge={self.judge.describe()} threshold={self.config.escalation_threshold} "
            f"history={self.config.escalation_history} "
            f"decisions={getattr(self.jev, 'endpoint', 'injected')}"
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

        if mode != MODE_SINGLE_TURN and config.enable_correction_escalation and clean_prompt:
            verdict = self.escalation_level(clean_prompt, context)
            if verdict is None:
                logger.warning("[pusula.route] escalation judge did not answer; staying on the agent default")
                return strong("judge_failed", fallback=True)
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
        if not self.jev.is_configured:
            return strong("jev_unconfigured", fallback=True)

        probability = self.jev.evaluate_boolean(
            state=clean_prompt,
            instructions=config.light_instructions or LIGHT_TURN_INSTRUCTIONS,
            timeout_ms=3000,
        )
        if not isinstance(probability, (int, float)):
            logger.warning("[pusula.route] Jev light-turn check failed; staying on the agent default")
            return strong("jev_failed", fallback=True)

        probability = float(probability)
        logger.info(f"[pusula.route] light-turn p={probability:.2f} (threshold {config.light_threshold})")
        if probability >= config.light_threshold:
            return decide(light, config.light_group, "light_turn", jev_calls=1, light_probability=probability)
        return strong("strong_turn", jev_calls=1, light_probability=probability)

    def escalation_level(
        self, prompt: str, context: dict[str, Any] | str | None
    ) -> tuple[int, tuple[str, ...]] | None:
        """Level 1 when this turn signals a miss; level 2 when the previous turn was a miss too.

        The judge answers two questions about this turn: is it a correction, and does it re-ask a
        recent request that did not end in `done`. Either one gives level 1. Level 2 needs the
        previous job to have ended blocked/unknown/failed, or the previous turn to have been a
        correction itself (one more question, asked only then). A plain failure with no user
        signal does not escalate: a provider outage is not a model problem.

        Returns None when the judge could not answer, so the caller can stay on the agent default.
        """
        config = self.config
        recent = _recent_jobs(context, config.escalation_window_seconds)
        deadline = self.judge.start_deadline()
        started = time.perf_counter()
        answers = self.judge.assess(prompt, recent, deadline=deadline)
        if answers is None:
            return None
        signals = [name for name in ("correction", "repeat") if answers.get(name, 0.0) >= config.escalation_threshold]
        summary = " ".join(f"{name}={answers[name]:.2f}" for name in sorted(answers))
        if not signals:
            logger.info(
                f"[pusula.route] judge {summary} (threshold {config.escalation_threshold}) "
                f"{(time.perf_counter() - started) * 1000:.0f} ms -> no escalation"
            )
            return 0, ()
        level = 1
        previous = recent[-1] if recent else None
        if previous is not None:
            repeated = _missed(previous)
            if not repeated:
                earlier = self.judge.is_correction(str(previous.get("transcript") or ""), recent[:-1], deadline=deadline)
                if earlier is None:
                    return None
                summary += f" previous_correction={earlier:.2f}"
                repeated = earlier >= config.escalation_threshold
            if repeated:
                level = 2
                signals.append("repeated_miss")
        logger.info(
            f"[pusula.route] judge {summary} (threshold {config.escalation_threshold}) "
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

    def _light_model(self) -> ModelEntry | None:
        if self.config.light_group == self.config.default_group:
            return None
        group = self.config.get_group(self.config.light_group)
        return group.primary() if group else None


def _judge_for(config: PusulaConfig) -> EscalationJudge:
    return EscalationJudge(
        url=config.escalation_judge_url,
        model=config.escalation_judge_model,
        timeout_ms=config.escalation_judge_timeout_ms,
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
