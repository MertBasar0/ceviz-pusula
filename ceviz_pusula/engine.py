from __future__ import annotations

import difflib
import logging
import re
import time
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard
from .jev_client import JevClient
from .model_catalog import ModelCatalog
from .pusula_types import (
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    ModelEntry,
    RouteDecision,
)

logger = logging.getLogger("ceviz.pusula.engine")

CORRECTION_PATTERN = re.compile(
    r"\b(anlamad[ıi]n|anlamam[ıi][sş]s[ıi]n|yanl[ıi][sş]|olmad[ıi]|tekrar dene|yeniden yap|"
    r"hata ald[ıi]m|hata verdi|ba[sş]ar[ıi]s[ıi]z|bunu demek istemedim|onu demedim|kastetmedim|"
    r"d[üu]zelt|tam tersi|eksik kald[ıi]|alakas[ıi]z|sa[cç]malad[ıi]n|[cç]al[ıi][sş]mad[ıi]|"
    r"i[sş]e yaramad[ıi]|yapamad[ıi]n|"
    r"not what i (?:meant|asked)|that'?s (?:wrong|not it)|(?:did|does)\s?n[o']?t work|try again)\b",
    re.IGNORECASE,
)

# A job that ended here did not get the user what they asked for.
MISSED_OUTCOMES = frozenset({"blocked", "unknown"})
REPEAT_SIMILARITY = 0.8
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

    Escalation is deterministic and stateless: it is derived from the transcript and the
    recent jobs Ceviz already persists, so a restart neither loses nor invents a level.
    """

    def __init__(
        self,
        state_dir: Path | str | None = None,
        jev_client: JevClient | None = None,
        catalog: ModelCatalog | None = None,
        agent: str = "main",
    ) -> None:
        self.config_guard = ConfigGuard(state_dir=state_dir)
        self.config, self.init_status = self.config_guard.load_config()
        self._jev_injected = jev_client is not None
        self.jev = jev_client or JevClient(system_one_url=self.config.decision_endpoint)
        self.catalog = catalog or ModelCatalog(agent, cache_path=self.config_guard.state_dir / "pusula-models.json")
        logger.info(f"[pusula] {self.describe()}")

    def refresh(self) -> None:
        """Reloads configuration."""
        self.config, self.init_status = self.config_guard.load_config()
        if not self._jev_injected:
            self.jev = JevClient(system_one_url=self.config.decision_endpoint)
        logger.info(f"[pusula] Refreshed: {self.describe()}")

    def describe(self) -> str:
        # No discovery here: `openclaw models list` runs only when a turn actually escalates.
        light = self._light_model()
        refs = self.config.escalation_models or ([self.config.escalation_model] if self.config.escalation_model else [])
        return (
            f"mode={self.config.routing_mode} strong=agent-default(with fallbacks) "
            f"light={light.id if light else None} light_threshold={self.config.light_threshold} "
            f"escalation={refs or 'auto(balanced)'}@{self.config.escalation_thinking} "
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

        if mode != MODE_SINGLE_TURN and config.enable_correction_escalation:
            level, signals = self.escalation_level(clean_prompt, context)
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

    def escalation_level(self, prompt: str, context: dict[str, Any] | str | None) -> tuple[int, tuple[str, ...]]:
        """Level 1 when this turn signals a miss; level 2 when the miss repeats inside the window.

        Signals on this turn: a correction phrase, or a near-repeat of a recent request that
        did not end in `done`. A repeat of the miss (an earlier correction in the window, or
        the previous job itself ending blocked/unknown) climbs one more rung. A plain failure
        with no user signal does not escalate: a provider outage is not a model problem.
        """
        signals: list[str] = []
        recent = _recent_jobs(context, self.config.escalation_window_seconds)
        if CORRECTION_PATTERN.search(prompt):
            signals.append("correction")
        normalized = _normalize(prompt)
        if normalized and any(
            (job.get("outcome") or "") != "done"
            and difflib.SequenceMatcher(None, normalized, _normalize(job.get("transcript") or "")).ratio()
            >= REPEAT_SIMILARITY
            for job in recent
        ):
            signals.append("repeat")
        if not signals:
            return 0, ()
        level = 1
        previous = recent[-1] if recent else None
        earlier_correction = any(CORRECTION_PATTERN.search(job.get("transcript") or "") for job in recent)
        if earlier_correction or (previous is not None and _missed(previous)):
            level = 2
            signals.append("repeated_miss")
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


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


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
