from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TIER_HEAVY_REMOTE = "heavy_remote"
TIER_LOW_LOCAL = "low_local"

# Downgrading a turn that needed tools costs a failed job; keeping a trivial turn on
# the strong model costs a few seconds. So the light tier needs a confident "yes".
DEFAULT_LIGHT_THRESHOLD = 0.8

DEFAULT_TIER_LABELS = {
    TIER_HEAVY_REMOTE: "Güçlü (varsayılan: araç kullanımı, iş devri, çok adımlı işler)",
    TIER_LOW_LOCAL: "Hafif (selamlaşma, sohbet, araç gerektirmeyen genel bilgi)",
}

DEFAULT_TIER_DESCRIPTIONS = {
    TIER_HEAVY_REMOTE: (
        "Default executor. Anything that may need tools, commands, files, live system state, "
        "GitHub, delegation to other agents, or earlier conversation."
    ),
    TIER_LOW_LOCAL: (
        "Small talk and general-knowledge questions that can be answered completely from memory "
        "without any tool."
    ),
}


@dataclass
class ModelEntry:
    id: str
    provider: str = ""
    name: str = ""
    thinking: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name,
            "thinking": self.thinking,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        return cls(
            id=str(data.get("id") or ""),
            provider=str(data.get("provider") or ""),
            name=str(data.get("name") or ""),
            thinking=data.get("thinking"),
            description=str(data.get("description") or ""),
        )


@dataclass
class TierGroup:
    name: str
    label: str = ""
    description: str = ""
    models: list[ModelEntry] = field(default_factory=list)
    primary_model: str | None = None

    def primary(self) -> ModelEntry | None:
        if not self.models:
            return None
        return next((m for m in self.models if m.id == self.primary_model), self.models[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "models": [m.to_dict() for m in self.models],
            "primary_model": self.primary_model,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> TierGroup:
        models_raw = data.get("models", [])
        models: list[ModelEntry] = []
        for item in models_raw:
            if isinstance(item, str):
                models.append(ModelEntry(id=item))
            elif isinstance(item, dict):
                models.append(ModelEntry.from_dict(item))

        return cls(
            name=name,
            label=str(data.get("label") or DEFAULT_TIER_LABELS.get(name, name)),
            description=str(data.get("description") or DEFAULT_TIER_DESCRIPTIONS.get(name, "")),
            models=models,
            primary_model=data.get("primary_model") or (models[0].id if models else None),
        )


RoutingMode = Literal["context_aware", "single_turn", "disabled"]
MODE_CONTEXT_AWARE = "context_aware"
MODE_SINGLE_TURN = "single_turn"
MODE_DISABLED = "disabled"


def _read_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LIGHT_THRESHOLD
    return min(max(threshold, 0.0), 1.0)


@dataclass
class PusulaConfig:
    enabled: bool = True
    routing_mode: str = MODE_CONTEXT_AWARE
    enable_correction_escalation: bool = True
    default_group: str = TIER_HEAVY_REMOTE
    default_model: str | None = None
    light_group: str = TIER_LOW_LOCAL
    light_threshold: float = DEFAULT_LIGHT_THRESHOLD
    escalation_model: str | None = None
    groups: dict[str, TierGroup] = field(default_factory=dict)

    def get_group(self, name: str) -> TierGroup | None:
        return self.groups.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "routing_mode": self.routing_mode,
            "enable_correction_escalation": self.enable_correction_escalation,
            "default_group": self.default_group,
            "default_model": self.default_model,
            "light_group": self.light_group,
            "light_threshold": self.light_threshold,
            "escalation_model": self.escalation_model,
            "groups": {name: g.to_dict() for name, g in self.groups.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PusulaConfig:
        groups_raw = data.get("groups", {})
        groups: dict[str, TierGroup] = {}
        if isinstance(groups_raw, dict):
            for name, g_data in groups_raw.items():
                if isinstance(g_data, dict):
                    groups[name] = TierGroup.from_dict(name, g_data)

        # Default routing_mode to context_aware if not explicitly set
        mode = str(data.get("routing_mode") or MODE_CONTEXT_AWARE).lower()
        if mode not in ("context_aware", "single_turn", "disabled"):
            mode = MODE_CONTEXT_AWARE

        enabled = bool(data.get("enabled", True))
        if mode == "disabled":
            enabled = False

        return cls(
            enabled=enabled,
            routing_mode=mode,
            enable_correction_escalation=bool(data.get("enable_correction_escalation", True)),
            default_group=str(data.get("default_group") or TIER_HEAVY_REMOTE),
            default_model=data.get("default_model"),
            light_group=str(data.get("light_group") or TIER_LOW_LOCAL),
            light_threshold=_read_threshold(data.get("light_threshold", DEFAULT_LIGHT_THRESHOLD)),
            escalation_model=data.get("escalation_model"),
            groups=groups,
        )


@dataclass
class RouteDecision:
    model: str | None
    group: str
    thinking: str | None = None
    jev_calls: int = 0
    latency_ms: float = 0.0
    reason: str = ""
    fallback: bool = False
    context_used: bool = False
    escalated: bool = False
    light_probability: float | None = None
