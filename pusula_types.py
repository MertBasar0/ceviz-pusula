from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TierGroupName = Literal["heavy_remote", "medium_remote", "low_local", str]
DecisionType = Literal["bypass", "boolean", "choice"]

DECISION_BYPASS = "bypass"
DECISION_BOOLEAN = "boolean"
DECISION_CHOICE = "choice"

TIER_HEAVY_REMOTE = "heavy_remote"
TIER_MEDIUM_REMOTE = "medium_remote"
TIER_LOW_LOCAL = "low_local"

DEFAULT_TIER_LABELS = {
    TIER_HEAVY_REMOTE: "Ağır ve Uzak (Derin Muhakeme, Mimari, Ağır Kodlama)",
    TIER_MEDIUM_REMOTE: "Orta ve Uzak (Bulut Hızlı Model, Genel Asistanlık, Belge/Özet)",
    TIER_LOW_LOCAL: "Düşük ve Yerel (Yerel Cihaz Modeli, Selamlaşma, Sistem Durumu)",
}

DEFAULT_TIER_DESCRIPTIONS = {
    TIER_HEAVY_REMOTE: (
        "Complex problems requiring deep multi-step reasoning, architectural coding, "
        "intricate debugging, advanced mathematical or algorithmic analysis."
    ),
    TIER_MEDIUM_REMOTE: (
        "Standard conversational queries, general knowledge lookup, summaries, routine API calls, "
        "and standard coding questions that do not need massive frontier models."
    ),
    TIER_LOW_LOCAL: (
        "Simple chit-chat, greetings, current time, date, local system status checks, "
        "or trivial queries that can be answered instantly by a small local/edge model."
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


@dataclass
class PusulaConfig:
    enabled: bool = True
    routing_mode: str = MODE_CONTEXT_AWARE
    enable_session_hysteresis: bool = True
    hysteresis_window_seconds: int = 900
    enable_correction_escalation: bool = True
    default_group: str = TIER_HEAVY_REMOTE
    default_model: str | None = None
    auto_diagnose_models: bool = True
    groups: dict[str, TierGroup] = field(default_factory=dict)

    def get_group(self, name: str) -> TierGroup | None:
        return self.groups.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "routing_mode": self.routing_mode,
            "enable_session_hysteresis": self.enable_session_hysteresis,
            "hysteresis_window_seconds": self.hysteresis_window_seconds,
            "enable_correction_escalation": self.enable_correction_escalation,
            "default_group": self.default_group,
            "default_model": self.default_model,
            "auto_diagnose_models": self.auto_diagnose_models,
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
            enable_session_hysteresis=bool(data.get("enable_session_hysteresis", True)),
            hysteresis_window_seconds=int(data.get("hysteresis_window_seconds", 900)),
            enable_correction_escalation=bool(data.get("enable_correction_escalation", True)),
            default_group=str(data.get("default_group") or TIER_HEAVY_REMOTE),
            default_model=data.get("default_model"),
            auto_diagnose_models=bool(data.get("auto_diagnose_models", True)),
            groups=groups,
        )


@dataclass
class ExecutionRecipe:
    stage1_decision: DecisionType
    stage1_options: list[str]
    stage1_instructions: str
    stage2_decisions: dict[str, DecisionType]
    summary: str


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
    hysteresis_applied: bool = False

