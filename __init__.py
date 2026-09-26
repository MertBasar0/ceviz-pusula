"""cevizPusula: keeps Ceviz turns on the strong executor unless a turn is clearly light."""

from .config_guard import ConfigGuard
from .engine import CevizPusula
from .jev_client import JevClient
from .pusula_types import (
    DEFAULT_LIGHT_THRESHOLD,
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
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
    "PusulaConfig",
    "TierGroup",
    "ModelEntry",
    "RouteDecision",
    "create_snapshot",
    "list_snapshots",
    "restore_snapshot",
    "set_routing_mode",
    "DEFAULT_LIGHT_THRESHOLD",
    "TIER_HEAVY_REMOTE",
    "TIER_LOW_LOCAL",
    "MODE_CONTEXT_AWARE",
    "MODE_SINGLE_TURN",
    "MODE_DISABLED",
]
