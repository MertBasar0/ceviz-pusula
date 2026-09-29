from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ceviz_pusula.config_guard import ConfigGuard
from ceviz_pusula.engine import CevizPusula
from ceviz_pusula.jev_client import JevClient
from ceviz_pusula.model_catalog import ModelCatalog, classify, rank
from ceviz_pusula.pusula_types import (
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    ModelEntry,
    TierGroup,
)
from ceviz_pusula.snapshot import create_snapshot, list_snapshots, restore_snapshot, set_routing_mode

LIGHT = "anthropic/claude-haiku-4-5"
SONNET = "anthropic/claude-sonnet-5"

# Shape of `openclaw models list --agent cevizmain --json` rows on the author's machine (2026-09-30).
CATALOG_ROWS = [
    {"key": "nvidia/nemotron-3-ultra-550b-a55b", "available": True, "tags": ["default", "configured"]},
    {"key": "nvidia/nemotron-3-super-120b-a12b", "available": True, "tags": ["configured"]},
    {"key": "anthropic/claude-haiku-4-5", "available": True, "tags": ["fallback#2", "configured"]},
    {"key": "anthropic/claude-opus-5", "available": True, "tags": ["configured"]},
    {"key": "anthropic/claude-opus-4-8", "available": True, "tags": ["configured"]},
    {"key": "anthropic/claude-sonnet-4-6", "available": True, "tags": ["configured"]},
    {"key": SONNET, "available": True, "tags": ["configured", "alias:sonnet"]},
    {"key": "openai/codex/gpt-5.5", "available": False, "tags": []},
    {"key": "openrouter/~anthropic/claude-sonnet-latest", "available": True, "tags": []},
    {"key": "vercel-ai-gateway/openai/gpt-5.6-mini", "available": True, "tags": []},
]


def fake_catalog(rows=CATALOG_ROWS) -> ModelCatalog:
    return ModelCatalog("cevizmain", runner=lambda: list(rows))


class TestConfigGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.guard = ConfigGuard(state_dir=self.state_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_creates_default_without_small_models(self) -> None:
        cfg, status = self.guard.load_config()
        self.assertEqual(status, "created_default")
        self.assertTrue(self.guard.config_path.is_file())
        self.assertEqual(set(cfg.groups), {TIER_HEAVY_REMOTE})
        self.assertEqual(cfg.routing_mode, MODE_CONTEXT_AWARE)
        self.assertEqual(cfg.escalation_models, [])
        self.assertEqual(cfg.escalation_thinking, ["medium", "high"])
        text = self.guard.config_path.read_text(encoding="utf-8")
        self.assertNotIn("120b", text)
        self.assertNotIn("30b", text)

    def test_recovers_from_corrupted_json_using_last_known_good(self) -> None:
        self.guard.load_config()
        self.guard.config_path.write_text("{ this is malformed json !!", encoding="utf-8")
        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_last_known_good")
        self.assertTrue(self.guard.broken_path.is_file())
        self.assertEqual(set(recovered_cfg.groups), {TIER_HEAVY_REMOTE})

    def test_recovers_to_builtin_if_lkg_also_corrupted(self) -> None:
        self.guard.config_path.write_text("invalid", encoding="utf-8")
        self.guard.last_known_good_path.write_text("invalid", encoding="utf-8")
        recovered_cfg, status = self.guard.load_config()
        self.assertEqual(status, "recovered_builtin_default")
        self.assertEqual(set(recovered_cfg.groups), {TIER_HEAVY_REMOTE})


class TestModelCatalog(unittest.TestCase):
    def test_classifies_frontier_families_and_ignores_others(self) -> None:
        self.assertEqual(classify("anthropic/claude-sonnet-5").role, "balanced")
        self.assertEqual(classify("anthropic/claude-sonnet-4-6").version, (4, 6))
        self.assertEqual(classify("anthropic/claude-haiku-4-5").role, "light")
        self.assertEqual(classify("anthropic/claude-opus-5").role, "frontier")
        self.assertEqual(classify("openai/gpt-5.6-mini").role, "balanced")
        self.assertEqual(classify("openai/gpt-5.5").role, "frontier")
        self.assertEqual(classify("google/gemini-3.1-flash-lite").role, "light")
        self.assertEqual(classify("google/gemini-3-pro").role, "frontier")
        self.assertIsNone(classify("nvidia/nemotron-3-super-120b-a12b"))

    def test_ranks_available_direct_models_newest_first(self) -> None:
        roles = rank(CATALOG_ROWS)
        self.assertEqual([m.ref for m in roles["balanced"]], [SONNET, "anthropic/claude-sonnet-4-6"])
        self.assertEqual(roles["frontier"][0].ref, "anthropic/claude-opus-5")
        self.assertEqual(roles["light"][0].ref, LIGHT)
        refs = {m.ref for ms in roles.values() for m in ms}
        self.assertNotIn("openai/codex/gpt-5.5", refs)  # available: false
        self.assertNotIn("openrouter/~anthropic/claude-sonnet-latest", refs)  # router, off by default

    def test_discovery_failure_falls_back_to_stale_cache_then_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "models.json"
            cache.write_text(json.dumps({"agent": "cevizmain", "fetched_at": 0, "models": CATALOG_ROWS}))

            def broken():
                raise RuntimeError("gateway down")

            self.assertEqual(ModelCatalog("cevizmain", cache, runner=broken).best("balanced").ref, SONNET)
            self.assertIsNone(ModelCatalog("cevizmain", Path(tmp) / "none.json", runner=broken).best("balanced"))


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
        return CevizPusula(state_dir=self.state_dir, jev_client=self.jev, catalog=fake_catalog(), agent="cevizmain")

    def _edit(self, **changes) -> None:
        guard = ConfigGuard(state_dir=self.state_dir)
        cfg, _ = guard.load_config()
        for key, value in changes.items():
            setattr(cfg, key, value)
        guard.save_config(cfg)

    def _with_light_tier(self) -> None:
        guard = ConfigGuard(state_dir=self.state_dir)
        cfg, _ = guard.load_config()
        cfg.groups[TIER_LOW_LOCAL] = TierGroup(name=TIER_LOW_LOCAL, models=[ModelEntry(id=LIGHT, thinking="low")])
        guard.save_config(cfg)

    @staticmethod
    def _job(transcript: str, outcome: str, ago: float = 60, status: str = "completed") -> dict:
        return {"transcript": transcript, "outcome": outcome, "status": status, "created_at": time.time() - ago}

    def test_default_config_keeps_every_plain_turn_on_the_agent_default(self) -> None:
        for text in ("Selam, nasılsın?", "PR 153915'i kontrol et"):
            decision = self._pusula().route(text)
            self.assertIsNone(decision.model)
            self.assertEqual(decision.reason, "no_light_tier")
        self.jev.evaluate_boolean.assert_not_called()

    def test_opt_in_light_tier_still_needs_a_confident_light_answer(self) -> None:
        self._with_light_tier()
        self.jev.evaluate_boolean.return_value = 0.93
        light = self._pusula().route("Selam, nasılsın?")
        self.assertEqual((light.model, light.thinking, light.reason), (LIGHT, "low", "light_turn"))
        self.jev.evaluate_boolean.return_value = 0.6
        self.assertIsNone(self._pusula().route("Kişisel mail kutuma erişimin var mı").model)

    def test_follow_up_in_active_task_skips_jev(self) -> None:
        self._with_light_tier()
        decision = self._pusula().route(
            "Onaylıyorum, başlayabilirsin",
            context={"continuation": "Önceki komut: Windows node kurulumunu tamamla"},
        )
        self.assertIsNone(decision.model)
        self.assertEqual(decision.reason, "active_context")
        self.jev.evaluate_boolean.assert_not_called()

    def test_correction_climbs_to_discovered_sonnet_at_medium(self) -> None:
        decision = self._pusula().route("Hayır bulutu kastetmedim. Claude subagent'ına işi devret")
        self.assertTrue(decision.escalated)
        self.assertEqual((decision.model, decision.thinking), (SONNET, "medium"))
        self.assertEqual((decision.escalation_level, decision.escalation_signals), (1, ("correction",)))
        self.assertEqual(decision.jev_calls, 0)

    def test_second_miss_in_window_climbs_to_high_thinking(self) -> None:
        context = {"recent_jobs": [
            self._job("Windows node'u kur", "needs_input", ago=300),
            self._job("Hayır, yanlış anladın, sen kur", "blocked", ago=120),
        ]}
        decision = self._pusula().route("Olmadı, tekrar dene", context=context)
        self.assertEqual((decision.model, decision.thinking, decision.escalation_level), (SONNET, "high", 2))
        self.assertIn("repeated_miss", decision.escalation_signals)

    def test_repeating_an_unanswered_request_escalates_without_a_correction_phrase(self) -> None:
        context = {"recent_jobs": [self._job("Claude Code oturumlarına erişebiliyor musun?", "needs_input")]}
        decision = self._pusula().route("Claude Code oturumlarına erişebiliyor musun", context=context)
        self.assertEqual(decision.escalation_level, 1)
        self.assertEqual(decision.escalation_signals, ("repeat",))

    def test_a_failed_job_alone_does_not_escalate_and_old_misses_expire(self) -> None:
        outage = {"recent_jobs": [self._job("PR'ı kontrol et", "unknown", status="failed")]}
        self.assertFalse(self._pusula().route("Selam", context=outage).escalated)
        stale = {"recent_jobs": [self._job("Windows node'u kur", "blocked", ago=3600)]}
        self.assertEqual(self._pusula().route("Olmadı", context=stale).escalation_level, 1)

    def test_correction_pattern_edges(self) -> None:
        pusula = self._pusula()
        self.assertFalse(pusula.route("Biraz önce 80'e çıkmış derken kastettiğim fail checklerde aslında.").escalated)
        self.assertTrue(pusula.route("That's not what I meant").escalated)
        self.assertTrue(pusula.route("Yine işe yaramadı").escalated)

    def test_configured_ladder_overrides_discovery(self) -> None:
        self._edit(escalation_models=["anthropic/claude-sonnet-4-6", "anthropic/claude-opus-5"],
                   escalation_thinking=["low", "high"])
        pusula = self._pusula()
        self.assertEqual([(m.id, m.thinking) for m in pusula.escalation_ladder()],
                         [("anthropic/claude-sonnet-4-6", "low"), ("anthropic/claude-opus-5", "high")])

    def test_no_discoverable_model_escalates_on_the_agent_default(self) -> None:
        pusula = CevizPusula(state_dir=self.state_dir, jev_client=self.jev,
                             catalog=fake_catalog(rows=CATALOG_ROWS[:2]), agent="cevizmain")
        decision = pusula.route("Yanlış yaptın")
        self.assertTrue(decision.escalated)
        self.assertIsNone(decision.model)

    def test_single_turn_mode_ignores_context_and_correction(self) -> None:
        self._with_light_tier()
        self._edit(routing_mode=MODE_SINGLE_TURN)
        self.jev.evaluate_boolean.return_value = 0.9
        decision = self._pusula().route("Yanlış yaptın, bunu düzelt", context={"continuation": "ağır iş"})
        self.assertFalse(decision.escalated)
        self.assertFalse(decision.context_used)
        self.assertEqual(decision.model, LIGHT)

    def test_disabled_mode_returns_no_override(self) -> None:
        self._edit(routing_mode=MODE_DISABLED)
        decision = self._pusula().route("Bunu yap")
        self.assertIsNone(decision.model)
        self.assertEqual(decision.reason, "pusula_disabled")

    def test_legacy_config_never_routes_to_middle_tier_and_ignores_group_escalation(self) -> None:
        legacy = {
            "enabled": True,
            "default_group": "heavy_remote",
            "groups": {
                "heavy_remote": {"models": [{"id": "nvidia/nemotron-3-ultra-550b-a55b"},
                                            {"id": LIGHT, "thinking": "high"}]},
                "medium_remote": {"models": [{"id": "nvidia/nemotron-3-super-120b-a12b"}]},
            },
        }
        (self.state_dir / "pusula.json").write_text(json.dumps(legacy), encoding="utf-8")
        pusula = self._pusula()
        self.assertIsNone(pusula.route("PR'ı kontrol et").model)
        self.assertEqual(pusula.route("Bu tamamen yanlış").model, SONNET)


class TestSystemOneBackend(unittest.TestCase):
    @staticmethod
    def _urlopen(body: dict) -> MagicMock:
        response = MagicMock()
        response.read.return_value = json.dumps(body).encode()
        opener = MagicMock()
        opener.return_value.__enter__.return_value = response
        return opener

    @staticmethod
    def _sent(opener: MagicMock) -> tuple:
        request = opener.call_args.args[0]
        return request.full_url, {k.lower(): v for k, v in request.header_items()}, json.loads(request.data)

    def test_local_system_one_boolean_uses_noul_without_credentials(self) -> None:
        opener = self._urlopen({"answers": {"eval_bool": {"type": "noul", "noul": 0.91}}})
        with patch("ceviz_pusula.jev_client.urllib.request.urlopen", opener):
            jev = JevClient(system_one_url="http://127.0.0.1:8009/")
            self.assertTrue(jev.is_configured)
            self.assertEqual(jev.evaluate_boolean(state="Selam", instructions="Small talk?"), 0.91)
        url, headers, body = self._sent(opener)
        self.assertEqual(url, "http://127.0.0.1:8009/v1/systemone")
        self.assertNotIn("authorization", headers)
        self.assertEqual(body["model"], "kev-latest")
        self.assertEqual(body["questions"]["eval_bool"]["type"], "noul")

    def test_http_error_and_timeout_return_none(self) -> None:
        import urllib.error
        jev = JevClient(system_one_url="http://127.0.0.1:8009")
        error = urllib.error.HTTPError("http://x", 403, "Forbidden", {}, None)
        with patch("ceviz_pusula.jev_client.urllib.request.urlopen", side_effect=error):
            self.assertIsNone(jev.evaluate_boolean(state="Selam", instructions="Small talk?"))
        with patch("ceviz_pusula.jev_client.urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertIsNone(jev.evaluate_boolean(state="Selam", instructions="Small talk?"))

    def test_light_tier_uses_configured_endpoint_question_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = ConfigGuard(state_dir=tmp)
            cfg, _ = guard.load_config()
            cfg.decision_endpoint = "http://127.0.0.1:8009"
            cfg.light_instructions = "Is this only small talk or a general-knowledge question?"
            cfg.light_threshold = 0.5
            cfg.groups[TIER_LOW_LOCAL] = TierGroup(name=TIER_LOW_LOCAL, models=[ModelEntry(id=LIGHT)])
            guard.save_config(cfg)
            opener = self._urlopen({"answers": {"eval_bool": {"type": "noul", "noul": 0.55}}})
            with patch("ceviz_pusula.jev_client.urllib.request.urlopen", opener):
                decision = CevizPusula(state_dir=tmp, catalog=fake_catalog()).route("Selam, nasılsın?")
        self.assertEqual((decision.reason, decision.model), ("light_turn", LIGHT))
        sent = self._sent(opener)[2]["questions"]["eval_bool"]
        self.assertEqual(sent["instructions"], "Is this only small talk or a general-knowledge question?")


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
        self.assertEqual([(s["name"], s["description"]) for s in snaps], [("test-snap", "Test snapshot")])

    def test_restore_snapshot(self) -> None:
        create_snapshot("baseline", state_dir=self.state_dir)
        set_routing_mode("single_turn", state_dir=self.state_dir)
        self.assertEqual(self.guard.load_config()[0].routing_mode, "single_turn")
        self.assertEqual(restore_snapshot("baseline", state_dir=self.state_dir).routing_mode, "context_aware")
        self.assertEqual(self.guard.load_config()[0].routing_mode, "context_aware")

    def test_mode_toggle(self) -> None:
        cfg = set_routing_mode("disabled", state_dir=self.state_dir)
        self.assertEqual((cfg.routing_mode, cfg.enabled), ("disabled", False))
        cfg = set_routing_mode("context_aware", state_dir=self.state_dir)
        self.assertEqual((cfg.routing_mode, cfg.enabled), ("context_aware", True))


if __name__ == "__main__":
    unittest.main()
