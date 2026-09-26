from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard
from .jev_client import JevClient
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
    r"d[üu]zelt|tam tersi|eksik kald[ıi]|alakas[ıi]z|sa[cç]malad[ıi]n)\b",
    re.IGNORECASE,
)

# Jev reads instructions literally and works best in English; the state stays the
# user's own (Turkish) words.
LIGHT_TURN_INSTRUCTIONS = (
    "Ceviz is a voice assistant that can run commands, read files, check GitHub, inspect this "
    "machine, and delegate work to other agents. Is this request only small talk or a "
    "general-knowledge question that can be answered completely from memory, without any tool, "
    "command, file, web lookup, live system state, earlier conversation, or delegation?"
)


class CevizPusula:
    """Keeps every turn on the agent's own model unless Jev is confident it is a light turn.

    Measured on 2026-09-26: mid-size models failed tool work (no delegation, write loops,
    unverified claims), while a wrong strong route only costs latency. So there is no
    middle tier and no multi-way classification: the only question is whether a fresh
    turn is clearly light.

    Strong turns carry no model override. OpenClaw disables the agent's configured
    fallback chain for any explicit `--model` run, so pinning the primary would turn
    one provider 503 into a failed job.
    """

    def __init__(
        self,
        state_dir: Path | str | None = None,
        jev_client: JevClient | None = None,
    ) -> None:
        self.config_guard = ConfigGuard(state_dir=state_dir)
        self.jev = jev_client or JevClient()
        self.config, self.init_status = self.config_guard.load_config()
        logger.info(f"[pusula] {self.describe()}")

    def refresh(self) -> None:
        """Reloads configuration."""
        self.config, self.init_status = self.config_guard.load_config()
        logger.info(f"[pusula] Refreshed: {self.describe()}")

    def describe(self) -> str:
        light = self._light_model()
        escalation = self._escalation_model()
        return (
            f"mode={self.config.routing_mode} strong=agent-default(with fallbacks) "
            f"light={light.id if light else None} light_threshold={self.config.light_threshold} "
            f"escalation={escalation.id if escalation else None}"
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

        if mode != MODE_SINGLE_TURN and config.enable_correction_escalation and CORRECTION_PATTERN.search(clean_prompt):
            escalation = self._escalation_model()
            logger.info(f"[pusula.route] Correction signal in '{clean_prompt[:40]}'; escalating to {escalation.id if escalation else 'agent default'}")
            return decide(escalation, config.default_group, "correction_escalation", escalated=True)

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
            instructions=LIGHT_TURN_INSTRUCTIONS,
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

    def _light_model(self) -> ModelEntry | None:
        if self.config.light_group == self.config.default_group:
            return None
        group = self.config.get_group(self.config.light_group)
        return group.primary() if group else None

    def _escalation_model(self) -> ModelEntry | None:
        group = self.config.get_group(self.config.default_group)
        models = group.models if group else []
        if self.config.escalation_model:
            return next(
                (m for m in models if m.id == self.config.escalation_model),
                ModelEntry(id=self.config.escalation_model),
            )
        return next((m for m in models if m.thinking), None)


def _has_context(context: dict[str, Any] | str | None) -> bool:
    if isinstance(context, str):
        return bool(context.strip())
    if isinstance(context, dict):
        return bool(str(context.get("continuation") or "").strip() or context.get("recent_job"))
    return False
