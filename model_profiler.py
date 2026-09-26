from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .jev_client import JevClient
from .pusula_types import (
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    TIER_MEDIUM_REMOTE,
    ModelEntry,
    PusulaConfig,
    TierGroup,
)

logger = logging.getLogger("ceviz.pusula.profiler")

DEFAULT_CACHE_FILE = Path.home() / ".openclaw" / "ceviz-state" / "pusula_model_cache.json"


class ModelProfiler:
    """Discovers available models and classifies them into inference tiers."""

    def __init__(
        self,
        cache_file: Path | str | None = None,
        jev_client: JevClient | None = None,
    ) -> None:
        self.cache_file = Path(cache_file) if cache_file else DEFAULT_CACHE_FILE
        self.jev_client = jev_client or JevClient()
        self.cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_file.is_file():
            try:
                data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"[pusula.profiler] Failed to read model cache: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[pusula.profiler] Failed to write model cache: {e}")

    def classify_model(self, model: ModelEntry | dict[str, Any]) -> str:
        """Classifies a model into heavy_remote, medium_remote, or low_local.

        Checks cache first. If missing, uses Jev or falls back to heuristics.
        """
        model_id = model.id if isinstance(model, ModelEntry) else str(model.get("id") or "")
        model_name = model.name if isinstance(model, ModelEntry) else str(model.get("name") or "")
        provider = model.provider if isinstance(model, ModelEntry) else str(model.get("provider") or "")
        thinking = model.thinking if isinstance(model, ModelEntry) else model.get("thinking")

        # 1. Check cache
        if model_id in self.cache:
            tier = self.cache[model_id].get("tier")
            if tier in (TIER_HEAVY_REMOTE, TIER_MEDIUM_REMOTE, TIER_LOW_LOCAL):
                return tier

        # 2. Try Jev evaluation if configured
        if self.jev_client and self.jev_client.is_configured:
            try:
                state_desc = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "provider": provider,
                    "thinking": thinking,
                }
                criteria = {
                    TIER_HEAVY_REMOTE: (
                        "Flagship frontier foundation models, models with high reasoning/thinking budget (e.g. Claude Opus, Haiku Thinking High, Nemotron 550B, DeepSeek-R1, GPT-5)."
                    ),
                    TIER_MEDIUM_REMOTE: (
                        "Fast cloud-hosted models for general coding, summaries, and standard queries (e.g. Nemotron 120B, Sonnet, GLM, Kimi, Minimax)."
                    ),
                    TIER_LOW_LOCAL: (
                        "Lightweight edge/local models, free tier low-latency models, or small parameter models (e.g. 8B models, Ollama local, free instant endpoints)."
                    ),
                }
                res = self.jev_client.evaluate_choice(
                    state=state_desc,
                    criteria=criteria,
                    instructions="Classify this AI model into its appropriate operational inference tier.",
                    timeout_ms=4000,
                )
                if res and res[0] in criteria:
                    chosen_tier = res[0]
                    logger.info(f"[pusula.profiler] Jev classified {model_id} -> {chosen_tier} (probs: {res[1]})")
                    self.cache[model_id] = {
                        "tier": chosen_tier,
                        "source": "jev",
                        "probabilities": res[1],
                    }
                    self._save_cache()
                    return chosen_tier
            except Exception as err:
                logger.warning(f"[pusula.profiler] Jev profiling error for {model_id}: {err}")

        # 3. Heuristic fallback
        tier = self._heuristic_classification(model_id, model_name, thinking, provider)
        logger.info(f"[pusula.profiler] Heuristics classified {model_id} -> {tier}")
        self.cache[model_id] = {
            "tier": tier,
            "source": "heuristic",
        }
        self._save_cache()
        return tier

    def _heuristic_classification(
        self,
        model_id: str,
        name: str,
        thinking: str | None,
        provider: str,
    ) -> str:
        s = f"{model_id} {name} {provider}".lower()

        # Local or small models
        if any(tok in s for tok in ["ollama", "localhost", "127.0.0.1", "local", ":free", "lightning"]):
            return TIER_LOW_LOCAL

        # Heavy frontier models
        if thinking and thinking.lower() in ("high", "medium"):
            return TIER_HEAVY_REMOTE

        if any(tok in s for tok in ["550b", "opus", "gpt-5", "deepseek-r1", "deepseek-v4", "ultra"]):
            return TIER_HEAVY_REMOTE

        # Medium default
        return TIER_MEDIUM_REMOTE

    @classmethod
    def discover_openclaw_models(cls, openclaw_config_path: Path | str | None = None) -> list[ModelEntry]:
        """Discovers configured models from OpenClaw's openclaw.json."""
        candidate_paths = [
            Path(openclaw_config_path) if openclaw_config_path else None,
            Path.home() / ".ocm" / "envs" / "nemo" / ".openclaw" / "openclaw.json",
            Path.home() / ".openclaw" / "openclaw.json",
        ]

        found_models: list[ModelEntry] = []
        for p in candidate_paths:
            if p and p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    providers = data.get("models", {}).get("providers", {})
                    for prov_id, prov_cfg in providers.items():
                        models_list = prov_cfg.get("models", [])
                        for m in models_list:
                            m_id = m.get("id")
                            if not m_id:
                                continue
                            m_name = m.get("name") or m_id
                            thinking = None
                            if m.get("params", {}).get("thinking"):
                                thinking = m["params"]["thinking"]
                            found_models.append(
                                ModelEntry(
                                    id=m_id,
                                    provider=prov_id,
                                    name=m_name,
                                    thinking=thinking,
                                    description=f"{prov_id} model: {m_name}",
                                )
                            )
                    if found_models:
                        break
                except Exception as e:
                    logger.warning(f"[pusula.profiler] Failed reading {p}: {e}")

        return found_models
