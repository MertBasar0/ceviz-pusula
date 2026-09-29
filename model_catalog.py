"""Discover the frontier models an OpenClaw agent can use and rank them into routing roles.

Source of truth is `openclaw models list --agent <id> --json`: it reflects the user's configured
providers, credentials and runtimes (for example Anthropic models served through the Claude CLI
subscription). Router catalogs (OpenRouter, Vercel AI Gateway) are ignored by default because
listing a model there says nothing about billing or free-tier access.

Roles:
  frontier  most capable, highest quota cost (Opus, GPT, Gemini Pro, Grok)
  balanced  strong reasoning at moderate quota cost (Sonnet, GPT mini, Gemini Flash)
  light     fastest and cheapest (Haiku, GPT nano, Gemini Flash-Lite, Grok mini)
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("ceviz.pusula.catalog")

ROLES = ("frontier", "balanced", "light")
DIRECT_PROVIDERS = ("anthropic", "openai", "google", "xai")
ROUTER_PROVIDERS = ("openrouter", "vercel-ai-gateway")
CACHE_TTL_SECONDS = 6 * 3600

# (provider, pattern on the model id after the provider prefix, family, role)
_FAMILY_RULES: list[tuple[str, re.Pattern[str], str, str]] = [
    ("anthropic", re.compile(r"^claude-opus-(\d+)(?:-(\d+))?"), "claude-opus", "frontier"),
    ("anthropic", re.compile(r"^claude-sonnet-(\d+)(?:-(\d+))?"), "claude-sonnet", "balanced"),
    ("anthropic", re.compile(r"^claude-haiku-(\d+)(?:-(\d+))?"), "claude-haiku", "light"),
    ("openai", re.compile(r"^(?:codex/)?gpt-(\d+)(?:\.(\d+))?-nano"), "gpt-nano", "light"),
    ("openai", re.compile(r"^(?:codex/)?gpt-(\d+)(?:\.(\d+))?-mini"), "gpt-mini", "balanced"),
    ("openai", re.compile(r"^(?:codex/)?gpt-(\d+)(?:\.(\d+))?$"), "gpt", "frontier"),
    ("google", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash-lite"), "gemini-flash-lite", "light"),
    ("google", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash"), "gemini-flash", "balanced"),
    ("google", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-pro"), "gemini-pro", "frontier"),
    ("xai", re.compile(r"^grok-(\d+)(?:\.(\d+))?-mini"), "grok-mini", "light"),
    ("xai", re.compile(r"^grok-(\d+)(?:\.(\d+))?$"), "grok", "frontier"),
]


@dataclass(frozen=True)
class CatalogModel:
    ref: str
    provider: str
    family: str
    role: str
    version: tuple[int, int]
    configured: bool


def classify(ref: str) -> CatalogModel | None:
    """Map a model reference to its frontier family and role, or None if it is not frontier."""
    provider, _, model_id = ref.partition("/")
    for rule_provider, pattern, family, role in _FAMILY_RULES:
        if provider != rule_provider:
            continue
        match = pattern.match(model_id)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return CatalogModel(ref, provider, family, role, (major, minor), False)
    return None


def _is_configured(tags: list[str]) -> bool:
    return any(tag == "configured" or tag == "default" or tag.startswith("fallback#") for tag in tags)


def rank(
    rows: list[dict[str, Any]],
    include_routers: bool = False,
    include_unconfigured: bool = False,
) -> dict[str, list[CatalogModel]]:
    """Group available frontier models by role, newest first.

    Only models the user already configured in OpenClaw are candidates by default: a merely
    available model can sit behind a pay-per-use API key the user never chose for Ceviz.
    """
    providers = DIRECT_PROVIDERS + (ROUTER_PROVIDERS if include_routers else ())
    by_role: dict[str, list[CatalogModel]] = {role: [] for role in ROLES}
    seen: set[str] = set()
    for row in rows:
        ref = str(row.get("key") or row.get("ref") or "")
        if not ref or ref in seen or row.get("available") is False:
            continue
        if ref.split("/", 1)[0] not in providers:
            continue
        configured = _is_configured(row.get("tags") or [])
        if not configured and not include_unconfigured:
            continue
        model = classify(ref)
        if model is None:
            continue
        seen.add(ref)
        by_role[model.role].append(CatalogModel(**{**model.__dict__, "configured": configured}))
    for role in ROLES:
        by_role[role].sort(key=lambda m: (m.configured, m.version), reverse=True)
    return by_role


class ModelCatalog:
    """Cached view of `openclaw models list` for one agent."""

    def __init__(self, agent: str, cache_path: Path | None = None, runner=None) -> None:
        self.agent = agent
        self.cache_path = cache_path
        self._runner = runner or self._run_openclaw
        self._rows: list[dict[str, Any]] | None = None

    def _run_openclaw(self) -> list[dict[str, Any]]:
        proc = subprocess.run(
            ["openclaw", "models", "list", "--agent", self.agent, "--json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        data = json.loads(proc.stdout or "{}")
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError((data.get("error") or {}).get("message", "unexpected models list output")
                             if isinstance(data, dict) else "unexpected models list output")
        return data["models"]

    def rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        cached = self._read_cache()
        if cached is not None:
            self._rows = cached
            return cached
        try:
            self._rows = self._runner()
            self._write_cache(self._rows)
        except Exception as exc:  # discovery must never break routing
            logger.warning(f"[pusula.catalog] model discovery failed: {exc}")
            self._rows = self._read_cache(allow_stale=True) or []
        return self._rows

    def roles(self, include_routers: bool = False) -> dict[str, list[CatalogModel]]:
        return rank(self.rows(), include_routers=include_routers)

    def best(self, role: str, include_routers: bool = False) -> CatalogModel | None:
        candidates = self.roles(include_routers).get(role) or []
        return candidates[0] if candidates else None

    def _read_cache(self, allow_stale: bool = False) -> list[dict[str, Any]] | None:
        if not self.cache_path or not self.cache_path.is_file():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            fresh = time.time() - float(data.get("fetched_at", 0)) < CACHE_TTL_SECONDS
            if data.get("agent") == self.agent and (fresh or allow_stale) and isinstance(data.get("models"), list):
                return data["models"]
        except Exception:
            return None
        return None

    def _write_cache(self, rows: list[dict[str, Any]]) -> None:
        if not self.cache_path:
            return
        slim = [{k: r.get(k) for k in ("key", "available", "tags")} for r in rows]
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"agent": self.agent, "fetched_at": time.time(), "models": slim}), encoding="utf-8")
        tmp.replace(self.cache_path)
