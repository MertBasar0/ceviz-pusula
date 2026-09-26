from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ceviz_pusula.config_guard import ConfigGuard
from ceviz_pusula.engine import CevizPusula
from ceviz_pusula.jev_client import JevClient
from ceviz_pusula.snapshot import create_snapshot, list_snapshots, restore_snapshot, set_routing_mode
from ceviz_pusula.pusula_types import (
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
)

STRONG = "nvidia/nemotron-3-ultra-550b-a55b"
LIGHT = "nvidia/nemotron-3.5-lightning-30b-a3b"
ESCALATION = "anthropic/claude-haiku-4-5"


class TestConfigGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.guard = ConfigGuard(state_dir=self.state_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_creates_default_when_missing(self) -> None:
        cfg, status = self.guard.load_config()
        self.assertEqual(status, "created_default")
        self.assertTrue(self.guard.config_path.is_file())
        self.assertTrue(self.guard.last_known_good_path.is_file())
        self.assertEqual(set(cfg.groups), {TIER_HEAVY_REMOTE, TIER_LOW_LOCAL})
        self.assertEqual(cfg.routing_mode, MODE_CONTEXT_AWARE)
        self.assertTrue(cfg.enable_correction_escalation)

    def test_recovers_from_corrupted_json_using_last_known_good(self) -> None:
        self.guard.load_config()
        self.guard.config_path.write_text("{ this is malformed json !!", encoding="utf-8")

        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_last_known_good")
        self.assertTrue(self.guard.broken_path.is_file())
        self.assertEqual(len(recovered_cfg.groups), 2)

    def test_recovers_to_builtin_if_lkg_also_corrupted(self) -> None:
        self.guard.config_path.write_text("invalid", encoding="utf-8")
        self.guard.last_known_good_path.write_text("invalid", encoding="utf-8")

        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_builtin_default")
        self.assertEqual(len(recovered_cfg.groups), 2)


class TestPusulaRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.jev = MagicMock(spec=JevClient)
        self.jev.is_configured = True
        self.jev.evaluate_boolean.return_value = 0.1

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _pusula(self) -> CevizPusula:
        return CevizPusula(state_dir=self.state_dir, jev_client=self.jev)

    def _set_mode(self, mode: str) -> None:
        guard = ConfigGuard(state_dir=self.state_dir)
        cfg, _ = guard.load_config()
        cfg.routing_mode = mode
        guard.save_config(cfg)

    def test_confident_light_turn_goes_to_light_model(self) -> None:
        self.jev.evaluate_boolean.return_value = 0.93
        decision = self._pusula().route("Selam, nasılsın?")

        self.assertEqual(decision.group, TIER_LOW_LOCAL)
        self.assertEqual(decision.model, LIGHT)
        self.assertEqual(decision.reason, "light_turn")
        self.assertEqual(decision.jev_calls, 1)
        self.assertEqual(decision.light_probability, 0.93)
        self.assertEqual(self.jev.evaluate_boolean.call_args.kwargs["state"], "Selam, nasılsın?")

    def test_unsure_turn_stays_on_strong_model(self) -> None:
        # Below the 0.8 threshold: a tool task must never be downgraded on a coin flip.
        self.jev.evaluate_boolean.return_value = 0.6
        decision = self._pusula().route("Kişisel mail kutuma erişimin var mı")

        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertEqual(decision.model, STRONG)
        self.assertEqual(decision.reason, "strong_turn")
        self.assertIsNone(decision.thinking)

    def test_jev_failure_stays_on_strong_model(self) -> None:
        self.jev.evaluate_boolean.return_value = None
        decision = self._pusula().route("Herhangi bir istek")

        self.assertTrue(decision.fallback)
        self.assertEqual(decision.model, STRONG)
        self.assertEqual(decision.reason, "jev_failed")

    def test_unconfigured_jev_stays_on_strong_model_without_calling_it(self) -> None:
        self.jev.is_configured = False
        decision = self._pusula().route("Selam")

        self.assertEqual(decision.model, STRONG)
        self.assertEqual(decision.reason, "jev_unconfigured")
        self.jev.evaluate_boolean.assert_not_called()

    def test_follow_up_in_active_task_skips_jev(self) -> None:
        self.jev.evaluate_boolean.return_value = 0.99
        decision = self._pusula().route(
            "Onaylıyorum, başlayabilirsin",
            context={"continuation": "Önceki komut: Windows node kurulumunu tamamla"},
        )

        self.assertEqual(decision.model, STRONG)
        self.assertEqual(decision.reason, "active_context")
        self.assertTrue(decision.context_used)
        self.jev.evaluate_boolean.assert_not_called()

    def test_recent_job_counts_as_active_context(self) -> None:
        decision = self._pusula().route(
            "Peki bu bahsedilen engelleri sen lokalde yapabilir misin?",
            context={"recent_job": {"transcript": "PR'daki son bot yorumlarına bak", "created_at": 0}},
        )
        self.assertEqual(decision.reason, "active_context")
        self.jev.evaluate_boolean.assert_not_called()

    def test_correction_escalates_to_thinking_model_without_jev(self) -> None:
        decision = self._pusula().route("Demek istediğimi anlamadın sanırım, pr ile ilgili işleri sen yap")

        self.assertTrue(decision.escalated)
        self.assertEqual(decision.reason, "correction_escalation")
        self.assertEqual(decision.model, ESCALATION)
        self.assertEqual(decision.thinking, "high")
        self.assertEqual(decision.jev_calls, 0)

    def test_correction_pattern_edges_from_09_23_transcripts(self) -> None:
        pusula = self._pusula()
        missed = pusula.route("Hayır bulutu kastetmedim. Claude subagent'ına işi devret")
        self.assertTrue(missed.escalated)

        false_positive = pusula.route("Biraz önce 80'e çıkmış derken kastettiğim fail checklerde aslında.")
        self.assertFalse(false_positive.escalated)

    def test_single_turn_mode_ignores_context_and_correction(self) -> None:
        self._set_mode(MODE_SINGLE_TURN)
        self.jev.evaluate_boolean.return_value = 0.9
        prompt = "Yanlış yaptın, bunu düzelt"
        decision = self._pusula().route(prompt, context={"continuation": "ağır iş"})

        self.assertFalse(decision.escalated)
        self.assertFalse(decision.context_used)
        self.assertEqual(decision.model, LIGHT)
        self.assertEqual(self.jev.evaluate_boolean.call_args.kwargs["state"], prompt)

    def test_disabled_mode_returns_no_override(self) -> None:
        self._set_mode(MODE_DISABLED)
        decision = self._pusula().route("Bunu yap")

        self.assertIsNone(decision.model)
        self.assertEqual(decision.reason, "pusula_disabled")

    def test_legacy_three_tier_config_never_routes_to_middle_tier(self) -> None:
        legacy = {
            "enabled": True,
            "routing_mode": "context_aware",
            "enable_session_hysteresis": True,
            "hysteresis_window_seconds": 900,
            "default_group": "heavy_remote",
            "default_model": STRONG,
            "groups": {
                "heavy_remote": {"models": [{"id": STRONG}, {"id": ESCALATION, "thinking": "high"}]},
                "medium_remote": {"models": [{"id": "nvidia/nemotron-3-super-120b-a12b"}]},
                "low_local": {"models": [{"id": LIGHT}]},
            },
        }
        (self.state_dir / "pusula.json").write_text(json.dumps(legacy), encoding="utf-8")
        pusula = self._pusula()

        for probability in (0.0, 0.5, 0.79):
            self.jev.evaluate_boolean.return_value = probability
            self.assertEqual(pusula.route("PR'ı kontrol et").model, STRONG)
        self.jev.evaluate_boolean.return_value = 0.95
        self.assertEqual(pusula.route("Selam").model, LIGHT)
        self.assertEqual(pusula.route("Bu tamamen yanlış").model, ESCALATION)


class TestSnapshotManager(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.guard = ConfigGuard(state_dir=self.state_dir)
        self.guard.load_config()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_and_list_snapshots(self) -> None:
        snap_path = create_snapshot("test-snap", description="Test snapshot", state_dir=self.state_dir)
        self.assertTrue(snap_path.is_file())

        snaps = list_snapshots(state_dir=self.state_dir)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["name"], "test-snap")
        self.assertEqual(snaps[0]["description"], "Test snapshot")

    def test_restore_snapshot(self) -> None:
        create_snapshot("baseline", state_dir=self.state_dir)

        set_routing_mode("single_turn", state_dir=self.state_dir)
        cfg, _ = self.guard.load_config()
        self.assertEqual(cfg.routing_mode, "single_turn")

        restored = restore_snapshot("baseline", state_dir=self.state_dir)
        self.assertEqual(restored.routing_mode, "context_aware")

        cfg_reloaded, _ = self.guard.load_config()
        self.assertEqual(cfg_reloaded.routing_mode, "context_aware")

    def test_mode_toggle(self) -> None:
        cfg = set_routing_mode("disabled", state_dir=self.state_dir)
        self.assertEqual(cfg.routing_mode, "disabled")
        self.assertFalse(cfg.enabled)

        cfg = set_routing_mode("context_aware", state_dir=self.state_dir)
        self.assertEqual(cfg.routing_mode, "context_aware")
        self.assertTrue(cfg.enabled)


if __name__ == "__main__":
    unittest.main()
