from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ceviz_pusula.config_guard import ConfigGuard, get_default_pusula_config
from ceviz_pusula.engine import CevizPusula
from ceviz_pusula.jev_client import JevClient
from ceviz_pusula.snapshot import create_snapshot, list_snapshots, restore_snapshot, set_routing_mode
from ceviz_pusula.topology_compiler import TopologyCompiler
from ceviz_pusula.pusula_types import (
    DECISION_BOOLEAN,
    DECISION_BYPASS,
    DECISION_CHOICE,
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    TIER_MEDIUM_REMOTE,
    ModelEntry,
    PusulaConfig,
    TierGroup,
)


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
        self.assertEqual(len(cfg.groups), 3)
        self.assertEqual(cfg.routing_mode, MODE_CONTEXT_AWARE)
        self.assertTrue(cfg.enable_session_hysteresis)
        self.assertTrue(cfg.enable_correction_escalation)

    def test_recovers_from_corrupted_json_using_last_known_good(self) -> None:
        # 1. Initial good load
        cfg, status = self.guard.load_config()
        self.assertEqual(status, "created_default")

        # 2. Corrupt pusula.json
        self.guard.config_path.write_text("{ this is malformed json !!", encoding="utf-8")

        # 3. Reload - must not raise!
        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_last_known_good")
        self.assertTrue(self.guard.broken_path.is_file())
        self.assertEqual(len(recovered_cfg.groups), 3)

    def test_recovers_to_builtin_if_lkg_also_corrupted(self) -> None:
        # Corrupt both
        self.guard.config_path.write_text("invalid", encoding="utf-8")
        self.guard.last_known_good_path.write_text("invalid", encoding="utf-8")

        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_builtin_default")
        self.assertEqual(len(recovered_cfg.groups), 3)


class TestTopologyCompiler(unittest.TestCase):
    def test_single_group_bypass(self) -> None:
        cfg = PusulaConfig(
            groups={
                "g1": TierGroup(name="g1", models=[ModelEntry(id="m1")]),
            }
        )
        recipe = TopologyCompiler.compile(cfg)
        self.assertEqual(recipe.stage1_decision, DECISION_BYPASS)
        self.assertEqual(recipe.stage2_decisions["g1"], DECISION_BYPASS)

    def test_two_groups_one_model_each_boolean_bypass(self) -> None:
        cfg = PusulaConfig(
            groups={
                "g1": TierGroup(name="g1", models=[ModelEntry(id="m1")]),
                "g2": TierGroup(name="g2", models=[ModelEntry(id="m2")]),
            }
        )
        recipe = TopologyCompiler.compile(cfg)
        self.assertEqual(recipe.stage1_decision, DECISION_BOOLEAN)
        self.assertEqual(recipe.stage2_decisions["g1"], DECISION_BYPASS)
        self.assertEqual(recipe.stage2_decisions["g2"], DECISION_BYPASS)

    def test_three_groups_choice_and_stage2_boolean(self) -> None:
        cfg = PusulaConfig(
            groups={
                "heavy": TierGroup(name="heavy", models=[ModelEntry(id="m1"), ModelEntry(id="m2")]),
                "medium": TierGroup(name="medium", models=[ModelEntry(id="m3")]),
                "low": TierGroup(name="low", models=[ModelEntry(id="m4")]),
            }
        )
        recipe = TopologyCompiler.compile(cfg)
        self.assertEqual(recipe.stage1_decision, DECISION_CHOICE)
        self.assertEqual(recipe.stage2_decisions["heavy"], DECISION_BOOLEAN)
        self.assertEqual(recipe.stage2_decisions["medium"], DECISION_BYPASS)
        self.assertEqual(recipe.stage2_decisions["low"], DECISION_BYPASS)


class TestPusulaEngineRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.mock_jev = MagicMock(spec=JevClient)
        self.mock_jev.is_configured = True
        self.mock_jev.evaluate_boolean.return_value = 0.8
        self.mock_jev.evaluate_choice.return_value = (TIER_HEAVY_REMOTE, {TIER_HEAVY_REMOTE: 1.0})

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_routes_to_low_when_choice_returns_low(self) -> None:
        self.mock_jev.evaluate_choice.return_value = (TIER_LOW_LOCAL, {TIER_LOW_LOCAL: 1.0})
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)

        decision = pusula.route("Selam")
        self.assertEqual(decision.group, TIER_LOW_LOCAL)
        self.assertEqual(decision.model, "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(decision.jev_calls, 1)

    def test_routes_to_heavy_and_evaluates_stage2_boolean(self) -> None:
        # Stage 1 choice returns heavy_remote
        self.mock_jev.evaluate_choice.return_value = (TIER_HEAVY_REMOTE, {TIER_HEAVY_REMOTE: 0.95})
        # Stage 2 boolean returns Claude Haiku (prob < 0.5)
        self.mock_jev.evaluate_boolean.return_value = 0.20

        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)
        decision = pusula.route("Kuantum denklemleri")

        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertEqual(decision.model, "anthropic/claude-haiku-4-5")
        self.assertEqual(decision.thinking, "high")
        self.assertEqual(decision.jev_calls, 2)

    def test_graceful_fallback_when_jev_fails(self) -> None:
        # Simulate network failure / timeout
        self.mock_jev.evaluate_choice.return_value = None
        self.mock_jev.evaluate_boolean.return_value = None

        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)
        decision = pusula.route("Herhangi bir istek")

        self.assertTrue(decision.fallback)
        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertIsNotNone(decision.model)

    def test_context_passed_to_jev_state_in_context_aware_mode(self) -> None:
        self.mock_jev.evaluate_choice.return_value = (TIER_HEAVY_REMOTE, {TIER_HEAVY_REMOTE: 0.96})
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)

        prompt = "Peki bu bahsedilen engelleri sen lokalde yapabilir misin?"
        context = {
            "continuation": "Önceki komut: PR bot yorumlarını analiz et\nÖnceki özet: CI testleri fail verdi",
        }
        decision = pusula.route(prompt, context=context)

        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertTrue(decision.context_used)

        # Verify Jev was called with structured state containing context
        call_kwargs = self.mock_jev.evaluate_choice.call_args[1]
        state_sent = call_kwargs.get("state")
        self.assertIsInstance(state_sent, dict)
        self.assertEqual(state_sent["current_user_request"], prompt)
        self.assertIn("PR bot yorumlarını analiz et", state_sent["recent_task_or_dialogue_context"])

    def test_correction_frustration_escalates_to_heavy(self) -> None:
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)

        # Frustrated user prompt
        prompt = "Demek istediğimi anlamadın sanırım. Demek istediğim pr ile ilgili yapılması gerekenleri sen yapabilir misin?"
        decision = pusula.route(prompt)

        # Must immediately escalate to heavy_remote without even needing Jev call
        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.reason, "correction_escalation")
        self.assertEqual(decision.jev_calls, 0)

    def test_session_hysteresis_maintains_heavy(self) -> None:
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)

        # Simulate that previous turn was heavy_remote 30 seconds ago
        context = {
            "last_tier": TIER_HEAVY_REMOTE,
            "last_tier_time": time.time() - 30,
            "summary": "GitHub PR incelemesi",
        }
        prompt = "Bunu hemen yapabilir misin?"
        decision = pusula.route(prompt, context=context)

        # Hysteresis locks to heavy_remote
        self.assertEqual(decision.group, TIER_HEAVY_REMOTE)
        self.assertTrue(decision.hysteresis_applied)
        self.assertEqual(decision.reason, "hysteresis_stickiness")

    def test_trivial_greeting_breaks_hysteresis(self) -> None:
        self.mock_jev.evaluate_choice.return_value = (TIER_LOW_LOCAL, {TIER_LOW_LOCAL: 0.99})
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)

        # Previous turn was heavy, but user says "Selam"
        context = {
            "last_tier": TIER_HEAVY_REMOTE,
            "last_tier_time": time.time() - 10,
        }
        decision = pusula.route("Selam, nasılsın?", context=context)

        self.assertFalse(decision.hysteresis_applied)
        self.assertEqual(decision.group, TIER_LOW_LOCAL)

    def test_single_turn_mode_bypasses_context_and_escalation(self) -> None:
        self.mock_jev.evaluate_choice.return_value = (TIER_MEDIUM_REMOTE, {TIER_MEDIUM_REMOTE: 0.8})
        guard = ConfigGuard(state_dir=self.state_dir)
        cfg, _ = guard.load_config()
        cfg.routing_mode = MODE_SINGLE_TURN
        guard.save_config(cfg)

        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)
        self.assertEqual(pusula.config.routing_mode, MODE_SINGLE_TURN)

        # Even with correction signal and context, single_turn must NOT escalate or pass dict state
        prompt = "Yanlış yaptın, bunu düzelt"
        context = {"summary": "heavy task"}
        decision = pusula.route(prompt, context=context)

        self.assertFalse(decision.escalated)
        self.assertFalse(decision.context_used)
        self.assertEqual(decision.group, TIER_MEDIUM_REMOTE)
        # Jev was called with raw string state
        call_kwargs = self.mock_jev.evaluate_choice.call_args[1]
        self.assertEqual(call_kwargs.get("state"), prompt)

    def test_disabled_mode_returns_none(self) -> None:
        guard = ConfigGuard(state_dir=self.state_dir)
        cfg, _ = guard.load_config()
        cfg.routing_mode = MODE_DISABLED
        guard.save_config(cfg)

        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.mock_jev)
        decision = pusula.route("Bunu yap")

        self.assertIsNone(decision.model)
        self.assertEqual(decision.reason, "pusula_disabled")


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
        # Create baseline with context_aware
        create_snapshot("baseline", state_dir=self.state_dir)

        # Modify active config to single_turn
        set_routing_mode("single_turn", state_dir=self.state_dir)
        cfg, _ = self.guard.load_config()
        self.assertEqual(cfg.routing_mode, "single_turn")

        # Restore baseline
        restored = restore_snapshot("baseline", state_dir=self.state_dir)
        self.assertEqual(restored.routing_mode, "context_aware")

        # Verify disk matches
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
