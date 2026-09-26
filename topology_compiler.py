from __future__ import annotations

import logging
from typing import Any

from .pusula_types import (
    DECISION_BOOLEAN,
    DECISION_BYPASS,
    DECISION_CHOICE,
    DEFAULT_TIER_DESCRIPTIONS,
    ExecutionRecipe,
    PusulaConfig,
    TierGroup,
)

logger = logging.getLogger("ceviz.pusula.topology")


class TopologyCompiler:
    """Compiles a PusulaConfig into an optimized execution recipe.

    Minimizes Jev LLM calls by dynamically compiling bypass, boolean,
    and choice stages based on the exact number of active groups and models.
    """

    @classmethod
    def compile(cls, config: PusulaConfig) -> ExecutionRecipe:
        active_groups = {k: v for k, v in config.groups.items() if v.models}
        group_keys = list(active_groups.keys())
        num_groups = len(group_keys)

        # Stage 1: Group selection decision
        if num_groups <= 1:
            stage1_decision = "bypass"
            stage1_options = group_keys
            stage1_instructions = "Only single group configured; bypass Jev."
        elif num_groups == 2:
            stage1_decision = "boolean"
            stage1_options = group_keys
            g1_name = group_keys[0]
            g1 = active_groups[g1_name]
            stage1_instructions = (
                f"Does the user prompt require {g1.label or g1_name}? "
                f"Context/Criteria: {g1.description or DEFAULT_TIER_DESCRIPTIONS.get(g1_name, '')}"
            )
        else:
            stage1_decision = "choice"
            stage1_options = group_keys
            stage1_instructions = (
                "Classify the user prompt into the most appropriate inference tier based on task complexity."
            )

        # Stage 2: Intra-group model selection decisions
        stage2_decisions: dict[str, str] = {}
        for g_name, group in active_groups.items():
            num_models = len(group.models)
            if num_models <= 1:
                stage2_decisions[g_name] = "bypass"
            elif num_models == 2:
                stage2_decisions[g_name] = "boolean"
            else:
                stage2_decisions[g_name] = "choice"

        # Build human-readable summary
        summary_lines = [
            f"Active Groups: {num_groups} ({', '.join(group_keys)})",
            f"Stage 1 Decision: {stage1_decision.upper()}",
        ]
        for g_name, dec in stage2_decisions.items():
            m_count = len(active_groups[g_name].models)
            summary_lines.append(f"Stage 2 [{g_name}]: {dec.upper()} ({m_count} models)")

        recipe = ExecutionRecipe(
            stage1_decision=stage1_decision,  # type: ignore[arg-type]
            stage1_options=stage1_options,
            stage1_instructions=stage1_instructions,
            stage2_decisions=stage2_decisions,  # type: ignore[arg-type]
            summary=" | ".join(summary_lines),
        )

        return recipe

    @classmethod
    def get_topology_receipt(cls, recipe: ExecutionRecipe, config: PusulaConfig) -> str:
        """Returns a formatted diagnostic receipt of the compiled topology."""
        lines = [
            "=" * 60,
            "              cevizPusula Topoloji Makbuzu",
            "=" * 60,
            f"Yapılandırma Durumu: {'AKTİF' if config.enabled else 'DEVRE DIŞI'}",
            f"Varsayılan Kademe: {config.default_group}",
            f"Tanımlı Grup Sayısı: {len(config.groups)}",
            "-" * 60,
            f"AŞAMA 1 (Kademe Seçimi): {recipe.stage1_decision.upper()}",
        ]

        if recipe.stage1_decision == "bypass":
            lines.append("  ↳ Karar: 0 Jev çağrısı (Tek grup veya bypass)")
        elif recipe.stage1_decision == "boolean":
            lines.append(f"  ↳ Karar: 1 Boolean Jev çağrısı ({recipe.stage1_options[0]} vs {recipe.stage1_options[1]})")
        elif recipe.stage1_decision == "choice":
            lines.append(f"  ↳ Karar: 1 Choice Jev çağrısı ({len(recipe.stage1_options)} seçenekli)")

        lines.append("-" * 60)
        lines.append("AŞAMA 2 (Grup İçi Model Seçimi):")

        total_models = 0
        min_calls = 0 if recipe.stage1_decision == "bypass" else 1
        max_calls = min_calls

        for g_name, g in config.groups.items():
            models = g.models
            total_models += len(models)
            s2_dec = recipe.stage2_decisions.get(g_name, "bypass")
            s2_str = f"  * {g_name} ({len(models)} model): {s2_dec.upper()}"
            if s2_dec == "bypass":
                s2_str += " ➔ 0 Jev çağrısı (Tek model / Sabit birincil)"
            elif s2_dec == "boolean":
                s2_str += f" ➔ 1 Boolean Jev çağrısı ({models[0].name or models[0].id} vs {models[1].name or models[1].id})"
            elif s2_dec == "choice":
                s2_str += f" ➔ 1 Choice Jev çağrısı ({len(models)} model arasında)"
            lines.append(s2_str)

        has_s2_call = any(d != "bypass" for d in recipe.stage2_decisions.values())
        if has_s2_call:
            max_calls += 1

        lines.append("-" * 60)
        lines.append(f"Toplam Model Sayısı: {total_models}")
        lines.append(f"Beklenen Jev Maliyeti / Gecikmesi: {min_calls} ila {max_calls} Jev çağrısı (~{min_calls*250}-{max_calls*250}ms)")
        lines.append("=" * 60)

        return "\n".join(lines)
