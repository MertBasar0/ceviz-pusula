from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .pusula_types import (
    DEFAULT_TIER_DESCRIPTIONS,
    DEFAULT_TIER_LABELS,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    TIER_MEDIUM_REMOTE,
    ModelEntry,
    PusulaConfig,
    TierGroup,
)

logger = logging.getLogger("ceviz.pusula.config")

DEFAULT_STATE_DIR = Path.home() / ".openclaw" / "ceviz-state"
CONFIG_FILENAME = "pusula.json"
LAST_KNOWN_GOOD_FILENAME = "pusula.last-known-good.json"
BROKEN_FILENAME = "pusula.broken.json"


def get_default_pusula_config() -> PusulaConfig:
    """Builds a robust, production-ready default configuration."""
    heavy_group = TierGroup(
        name=TIER_HEAVY_REMOTE,
        label=DEFAULT_TIER_LABELS[TIER_HEAVY_REMOTE],
        description=DEFAULT_TIER_DESCRIPTIONS[TIER_HEAVY_REMOTE],
        models=[
            ModelEntry(
                id="nvidia/nemotron-3-ultra-550b-a55b",
                provider="nvidia",
                name="Nemotron 3 Ultra 550B",
                description="Flagship 550B frontier model for complex tasks and deep reasoning.",
            ),
            ModelEntry(
                id="anthropic/claude-haiku-4-5",
                provider="claude-cli",
                name="Claude Haiku 4.5 (Thinking High)",
                thinking="high",
                description="Fast Claude model with high reasoning/thinking budget.",
            ),
        ],
        primary_model="nvidia/nemotron-3-ultra-550b-a55b",
    )

    medium_group = TierGroup(
        name=TIER_MEDIUM_REMOTE,
        label=DEFAULT_TIER_LABELS[TIER_MEDIUM_REMOTE],
        description=DEFAULT_TIER_DESCRIPTIONS[TIER_MEDIUM_REMOTE],
        models=[
            ModelEntry(
                id="nvidia/nemotron-3-super-120b-a12b",
                provider="nvidia",
                name="Nemotron 3 Super 120B",
                description="Fast balanced 120B cloud model for general conversation.",
            ),
        ],
        primary_model="nvidia/nemotron-3-super-120b-a12b",
    )

    low_group = TierGroup(
        name=TIER_LOW_LOCAL,
        label=DEFAULT_TIER_LABELS[TIER_LOW_LOCAL],
        description=DEFAULT_TIER_DESCRIPTIONS[TIER_LOW_LOCAL],
        models=[
            ModelEntry(
                id="nvidia/nemotron-3.5-lightning-30b-a3b",
                provider="nvidia",
                name="Nemotron 3.5 Lightning 30B",
                description="Instant, lightweight model for casual greetings, simple questions, and quick status.",
            ),
        ],
        primary_model="nvidia/nemotron-3.5-lightning-30b-a3b",
    )

    return PusulaConfig(
        enabled=True,
        routing_mode="context_aware",
        enable_session_hysteresis=True,
        hysteresis_window_seconds=900,
        enable_correction_escalation=True,
        default_group=TIER_HEAVY_REMOTE,
        default_model="nvidia/nemotron-3-ultra-550b-a55b",
        auto_diagnose_models=True,
        groups={
            TIER_HEAVY_REMOTE: heavy_group,
            TIER_MEDIUM_REMOTE: medium_group,
            TIER_LOW_LOCAL: low_group,
        },
    )


class ConfigGuard:
    """Self-healing, zero-crash configuration manager for cevizPusula."""

    def __init__(self, state_dir: Path | str | None = None) -> None:
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.config_path = self.state_dir / CONFIG_FILENAME
        self.last_known_good_path = self.state_dir / LAST_KNOWN_GOOD_FILENAME
        self.broken_path = self.state_dir / BROKEN_FILENAME

    def load_config(self) -> tuple[PusulaConfig, str]:
        """Loads configuration with self-healing guarantees.

        Returns:
            (config, status_code)
            status_code is one of:
            - 'loaded_ok'
            - 'created_default'
            - 'recovered_last_known_good'
            - 'recovered_builtin_default'
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_path.is_file():
            logger.info(f"[pusula.config] Configuration file not found at {self.config_path}; generating default template.")
            default_cfg = get_default_pusula_config()
            self._save_file_atomic(self.config_path, default_cfg.to_dict())
            self._save_file_atomic(self.last_known_good_path, default_cfg.to_dict())
            return default_cfg, "created_default"

        # Try to parse existing config
        raw_text = ""
        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Root JSON element must be an object/dict")

            config = PusulaConfig.from_dict(data)

            # Ensure minimal validity
            if not config.groups:
                raise ValueError("Configuration has no tier groups defined")

            # Successfully loaded and valid; update last known good
            self._save_file_atomic(self.last_known_good_path, data)
            return config, "loaded_ok"

        except Exception as exc:
            logger.warning(
                f"[pusula.config] CRITICAL: Failed to parse user config at {self.config_path}: {exc}. "
                f"Activating self-healing recovery."
            )
            # Backup the broken file so user doesn't lose their edits
            try:
                if self.config_path.is_file():
                    shutil.copy2(self.config_path, self.broken_path)
                    logger.info(f"[pusula.config] Preserved corrupted file at {self.broken_path}")
            except Exception as backup_err:
                logger.error(f"[pusula.config] Failed to preserve broken config: {backup_err}")

            # Step 1: Try last-known-good
            if self.last_known_good_path.is_file():
                try:
                    lkg_text = self.last_known_good_path.read_text(encoding="utf-8")
                    lkg_data = json.loads(lkg_text)
                    if isinstance(lkg_data, dict):
                        lkg_config = PusulaConfig.from_dict(lkg_data)
                        if lkg_config.groups:
                            logger.info("[pusula.config] Successfully restored from last-known-good configuration.")
                            return lkg_config, "recovered_last_known_good"
                except Exception as lkg_err:
                    logger.warning(f"[pusula.config] Last known good config was also invalid: {lkg_err}")

            # Step 2: Fall back to safe built-in default
            logger.warning("[pusula.config] Falling back to built-in safe default configuration.")
            fallback_cfg = get_default_pusula_config()
            return fallback_cfg, "recovered_builtin_default"

    def save_config(self, config: PusulaConfig) -> bool:
        """Saves configuration safely and updates last-known-good."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            data = config.to_dict()
            self._save_file_atomic(self.config_path, data)
            self._save_file_atomic(self.last_known_good_path, data)
            logger.info(f"[pusula.config] Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"[pusula.config] Error saving configuration: {e}")
            return False

    def _save_file_atomic(self, target_path: Path, data: dict[str, Any]) -> None:
        temp_path = target_path.with_suffix(".tmp")
        content = json.dumps(data, indent=2, ensure_ascii=False)
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(target_path)
