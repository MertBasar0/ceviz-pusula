"""cevizPusula: Intelligent and Adaptive Inference Router for Ceviz and OpenClaw."""

from .config_guard import ConfigGuard
from .engine import CevizPusula
from .jev_client import JevClient
from .model_profiler import ModelProfiler
from .topology_compiler import TopologyCompiler
from .pusula_types import (
    DECISION_BOOLEAN,
    DECISION_BYPASS,
    DECISION_CHOICE,
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


def __getattr__(name: str):
    if name in ("create_snapshot", "list_snapshots", "restore_snapshot", "set_routing_mode"):
        from . import snapshot
        return getattr(snapshot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CevizPusula",
    "ConfigGuard",
    "JevClient",
    "ModelProfiler",
    "TopologyCompiler",
    "PusulaConfig",
    "TierGroup",
    "ModelEntry",
    "ExecutionRecipe",
    "RouteDecision",
    "create_snapshot",
    "list_snapshots",
    "restore_snapshot",
    "set_routing_mode",
    "TIER_HEAVY_REMOTE",
    "TIER_MEDIUM_REMOTE",
    "TIER_LOW_LOCAL",
    "DECISION_BYPASS",
    "DECISION_BOOLEAN",
    "DECISION_CHOICE",
    "MODE_CONTEXT_AWARE",
    "MODE_SINGLE_TURN",
    "MODE_DISABLED",
]
