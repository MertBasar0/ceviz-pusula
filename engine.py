from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard
from .jev_client import JevClient
from .model_profiler import ModelProfiler
from .topology_compiler import TopologyCompiler
from .pusula_types import (
    DECISION_BOOLEAN,
    DECISION_BYPASS,
    DECISION_CHOICE,
    DEFAULT_TIER_DESCRIPTIONS,
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    TIER_MEDIUM_REMOTE,
    ExecutionRecipe,
    ModelEntry,
    PusulaConfig,
    RouteDecision,
    TierGroup,
)

logger = logging.getLogger("ceviz.pusula.engine")

CORRECTION_PATTERN = re.compile(
    r"\b(anlamad[ıi]n|anlamam[ıi][sş]s[ıi]n|yanl[ıi][sş]|olmad[ıi]|tekrar dene|yeniden yap|"
    r"hata ald[ıi]m|hata verdi|ba[sş]ar[ıi]s[ıi]z|bunu demek istemedim|kastetti[gğ]im|"
    r"d[üu]zelt|tam tersi|eksik kald[ıi]|alakas[ıi]z|sa[cç]malad[ıi]n)\b",
    re.IGNORECASE,
)

TRIVIAL_GREETING_PATTERN = re.compile(
    r"^(selam|merhaba|g[üu]nayd[ıi]n|iyi ak[sş]amlar|iyi geceler|hey|saat ka[cç]|hava nas[ıi]l|"
    r"nas[ıi]ls[ıi]n|kimsin|te[sş]ekk[üu]rler|sa[gğ] ol|tamamd[ıi]r)\b",
    re.IGNORECASE,
)


class CevizPusula:
    """Intelligent semantic model router for Ceviz and OpenClaw."""

    def __init__(
        self,
        state_dir: Path | str | None = None,
        jev_client: JevClient | None = None,
    ) -> None:
        self.config_guard = ConfigGuard(state_dir=state_dir)
        self.jev = jev_client or JevClient()
        self.profiler = ModelProfiler(jev_client=self.jev)
        self.config, self.init_status = self.config_guard.load_config()
        self.recipe = TopologyCompiler.compile(self.config)

        # Dynamic session tracking for hysteresis
        self._last_routed_tier: str | None = None
        self._last_routed_time: float = 0.0
        self._last_routed_model: str | None = None

        # Print initial diagnostic receipt
        receipt = TopologyCompiler.get_topology_receipt(self.recipe, self.config)
        logger.info(f"\n{receipt}")

    def refresh(self) -> None:
        """Reloads configuration and recompiles topology."""
        self.config, self.init_status = self.config_guard.load_config()
        self.recipe = TopologyCompiler.compile(self.config)
        receipt = TopologyCompiler.get_topology_receipt(self.recipe, self.config)
        logger.info(f"[pusula] Refreshed topology:\n{receipt}")

    def route(
        self,
        prompt: str,
        context: dict[str, Any] | str | None = None,
    ) -> RouteDecision:
        """Evaluates a user prompt and routes it to the optimal model and inference tier."""
        start_time = time.perf_counter()
        clean_prompt = (prompt or "").strip()

        # 1. Check if Pusula is enabled or in disabled mode
        mode = getattr(self.config, "routing_mode", MODE_CONTEXT_AWARE)
        if not self.config.enabled or mode == MODE_DISABLED:
            return RouteDecision(
                model=None,
                group=self.config.default_group,
                thinking=None,
                jev_calls=0,
                latency_ms=0.0,
                reason="pusula_disabled",
                fallback=False,
            )

        # 2. Check if Jev is configured
        if not self.jev.is_configured:
            default_grp = self.config.get_group(self.config.default_group)
            model_id = self.config.default_model or (default_grp.primary_model if default_grp else None)
            return RouteDecision(
                model=model_id,
                group=self.config.default_group,
                thinking=None,
                jev_calls=0,
                latency_ms=0.0,
                reason="jev_unconfigured_default",
                fallback=True,
            )

        # 3. Context & Feature extraction based on mode
        is_single_turn = (mode == MODE_SINGLE_TURN)
        context_text = ""
        last_tier = self._last_routed_tier
        last_tier_time = self._last_routed_time

        if not is_single_turn and context:
            if isinstance(context, str):
                context_text = context.strip()
            elif isinstance(context, dict):
                parts = []
                cont = context.get("continuation")
                if cont:
                    parts.append(str(cont).strip())
                recent = context.get("recent_job")
                if isinstance(recent, dict):
                    q = recent.get("transcript") or recent.get("name")
                    summary = recent.get("watch_summary") or recent.get("canned_result")
                    if q:
                        parts.append(f"Önceki iş: {q}")
                    if summary:
                        parts.append(f"Önceki özet: {summary}")
                elif isinstance(recent, str) and recent.strip():
                    parts.append(recent.strip())

                if context.get("last_tier"):
                    last_tier = str(context["last_tier"])
                if context.get("last_tier_time"):
                    try:
                        last_tier_time = float(context["last_tier_time"])
                    except (ValueError, TypeError):
                        pass

                context_text = "\n".join(parts).strip()

        # 4. Correction / Frustration Escalation
        apply_escalation = getattr(self.config, "enable_correction_escalation", True) and not is_single_turn
        if apply_escalation and CORRECTION_PATTERN.search(clean_prompt):
            logger.info(f"[pusula.route] Correction signal detected in '{clean_prompt[:40]}'; escalating to {TIER_HEAVY_REMOTE}")
            heavy_grp = self.config.get_group(TIER_HEAVY_REMOTE)
            if heavy_grp and heavy_grp.models:
                # Prefer model with thinking if available, else primary
                selected_model = next((m for m in heavy_grp.models if m.thinking == "high"), None)
                if not selected_model:
                    selected_model = next((m for m in heavy_grp.models if m.id == heavy_grp.primary_model), heavy_grp.models[0])
                elapsed = (time.perf_counter() - start_time) * 1000
                self._record_routing(TIER_HEAVY_REMOTE, selected_model.id)
                return RouteDecision(
                    model=selected_model.id,
                    group=TIER_HEAVY_REMOTE,
                    thinking=selected_model.thinking,
                    jev_calls=0,
                    latency_ms=round(elapsed, 1),
                    reason="correction_escalation",
                    fallback=False,
                    context_used=bool(context_text),
                    escalated=True,
                    hysteresis_applied=False,
                )

        # 5. Session Tier Stickiness / Hysteresis
        apply_hysteresis = getattr(self.config, "enable_session_hysteresis", True) and not is_single_turn
        hysteresis_applied = False
        selected_group_name = self.config.default_group
        locked_by_hysteresis = False

        if apply_hysteresis and last_tier == TIER_HEAVY_REMOTE:
            window = getattr(self.config, "hysteresis_window_seconds", 900)
            now = time.time()
            if (now - last_tier_time) <= window and not TRIVIAL_GREETING_PATTERN.search(clean_prompt):
                logger.info(
                    f"[pusula.route] Session hysteresis active ({int(now - last_tier_time)}s <= {window}s in {last_tier}); "
                    f"maintaining {TIER_HEAVY_REMOTE}"
                )
                selected_group_name = TIER_HEAVY_REMOTE
                locked_by_hysteresis = True
                hysteresis_applied = True

        jev_calls = 0
        fallback = False

        # Prepare state for Jev
        if not is_single_turn and context_text:
            eval_state: Any = {
                "current_user_request": clean_prompt,
                "recent_task_or_dialogue_context": context_text[:1200],
            }
        else:
            eval_state = clean_prompt

        # --- STAGE 1: Inference Tier Selection ---
        active_groups = {k: v for k, v in self.config.groups.items() if v.models}
        s1_type = self.recipe.stage1_decision

        if not locked_by_hysteresis:
            if s1_type == DECISION_BYPASS:
                if self.recipe.stage1_options:
                    selected_group_name = self.recipe.stage1_options[0]
            elif s1_type == DECISION_BOOLEAN:
                # 2 groups: boolean question
                g1_name = self.recipe.stage1_options[0]
                g2_name = self.recipe.stage1_options[1]
                prob = self.jev.evaluate_boolean(
                    state=eval_state,
                    instructions=self.recipe.stage1_instructions,
                    timeout_ms=3000,
                )
                if isinstance(prob, (int, float)):
                    jev_calls += 1
                    selected_group_name = g1_name if prob >= 0.5 else g2_name
                    logger.info(f"[pusula.route] Stage 1 Boolean: {g1_name} prob={prob:.2f} -> chose {selected_group_name}")
                else:
                    fallback = True
                    selected_group_name = self.config.default_group
                    logger.warning(f"[pusula.route] Stage 1 Boolean failed, falling back to {selected_group_name}")

            elif s1_type == DECISION_CHOICE:
                # 3+ groups: multiple choice
                criteria: dict[str, str] = {}
                for g_name in self.recipe.stage1_options:
                    grp = active_groups.get(g_name)
                    desc = (grp.description if grp and grp.description else DEFAULT_TIER_DESCRIPTIONS.get(g_name, ""))
                    label = grp.label if grp and grp.label else g_name
                    criteria[g_name] = f"{label}: {desc}"

                res = self.jev.evaluate_choice(
                    state=eval_state,
                    criteria=criteria,
                    instructions=self.recipe.stage1_instructions,
                    timeout_ms=3000,
                )
                if res and res[0] in criteria:
                    jev_calls += 1
                    selected_group_name = res[0]
                    logger.info(f"[pusula.route] Stage 1 Choice: chose {selected_group_name} (probs: {res[1]})")
                else:
                    fallback = True
                    selected_group_name = self.config.default_group
                    logger.warning(f"[pusula.route] Stage 1 Choice failed, falling back to {selected_group_name}")

        # Ensure chosen group is valid
        chosen_group = self.config.get_group(selected_group_name)
        if not chosen_group or not chosen_group.models:
            chosen_group = self.config.get_group(self.config.default_group)
            selected_group_name = self.config.default_group
            fallback = True

        if not chosen_group or not chosen_group.models:
            elapsed = (time.perf_counter() - start_time) * 1000
            return RouteDecision(
                model=self.config.default_model,
                group=selected_group_name,
                thinking=None,
                jev_calls=jev_calls,
                latency_ms=elapsed,
                reason="empty_group_fallback",
                fallback=True,
                context_used=bool(context_text),
                escalated=False,
                hysteresis_applied=hysteresis_applied,
            )

        # --- STAGE 2: Intra-Group Model Selection ---
        s2_type = self.recipe.stage2_decisions.get(selected_group_name, DECISION_BYPASS)
        selected_model: ModelEntry | None = None

        if s2_type == DECISION_BYPASS or len(chosen_group.models) <= 1:
            primary_id = chosen_group.primary_model
            selected_model = next((m for m in chosen_group.models if m.id == primary_id), chosen_group.models[0])

        elif s2_type == DECISION_BOOLEAN and len(chosen_group.models) == 2:
            m1 = chosen_group.models[0]
            m2 = chosen_group.models[1]
            instructions = (
                f"Between these two models, does this user query specifically require the heavy/higher reasoning capability of "
                f"'{m1.name or m1.id}' ({m1.description or 'advanced reasoning'}) rather than '{m2.name or m2.id}' ({m2.description or 'standard'})?"
            )
            prob = self.jev.evaluate_boolean(state=eval_state, instructions=instructions, timeout_ms=2500)
            if isinstance(prob, (int, float)):
                jev_calls += 1
                selected_model = m1 if prob >= 0.5 else m2
                logger.info(f"[pusula.route] Stage 2 Boolean: {m1.name} prob={prob:.2f} -> chose {selected_model.id}")
            else:
                fallback = True
                selected_model = next((m for m in chosen_group.models if m.id == chosen_group.primary_model), m1)

        elif s2_type == DECISION_CHOICE:
            criteria = {}
            for m in chosen_group.models:
                criteria[m.id] = f"{m.name or m.id}: {m.description or 'Inference model'}"

            res = self.jev.evaluate_choice(
                state=eval_state,
                criteria=criteria,
                instructions="Select the single model that is best suited to fulfill this request.",
                timeout_ms=2500,
            )
            if res and res[0] in criteria:
                jev_calls += 1
                choice_id = res[0]
                selected_model = next((m for m in chosen_group.models if m.id == choice_id), None)
                logger.info(f"[pusula.route] Stage 2 Choice: chose {choice_id} (probs: {res[1]})")
            else:
                fallback = True
                selected_model = next(
                    (m for m in chosen_group.models if m.id == chosen_group.primary_model),
                    chosen_group.models[0],
                )

        if not selected_model:
            selected_model = chosen_group.models[0]

        self._record_routing(selected_group_name, selected_model.id)

        elapsed = (time.perf_counter() - start_time) * 1000
        reason = "hysteresis_stickiness" if locked_by_hysteresis else f"routed_to_{selected_group_name}"
        return RouteDecision(
            model=selected_model.id,
            group=selected_group_name,
            thinking=selected_model.thinking,
            jev_calls=jev_calls,
            latency_ms=round(elapsed, 1),
            reason=reason,
            fallback=fallback,
            context_used=bool(context_text),
            escalated=False,
            hysteresis_applied=hysteresis_applied,
        )

    def _record_routing(self, group: str, model_id: str) -> None:
        self._last_routed_tier = group
        self._last_routed_time = time.time()
        self._last_routed_model = model_id
