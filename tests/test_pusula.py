from __future__ import annotations

import io
import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ceviz_pusula.config_guard import ConfigGuard
from ceviz_pusula.engine import CevizPusula
from ceviz_pusula.escalation_judge import QUESTIONS, EscalationJudge, build_prompt, build_state, letter_probability
from ceviz_pusula.jev_client import JevClient
from ceviz_pusula.model_catalog import ModelCatalog, classify, rank
from ceviz_pusula.pusula_types import (
    MODE_CONTEXT_AWARE,
    MODE_DISABLED,
    MODE_SINGLE_TURN,
    TIER_HEAVY_REMOTE,
    TIER_LOW_LOCAL,
    ModelEntry,
    PusulaConfig,
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


class FakeJudge:
    """Stands in for the decision model: fixed P(yes) per message, everything else a clear no."""

    QUIET = {"correction": 0.04, "repeat": 0.06}

    def __init__(self) -> None:
        self.answers: dict[str, dict[str, float]] = {}
        self.previous: dict[str, float] = {}
        self.calls: list[tuple] = []
        self.fail = False
        self.fail_previous = False

    def describe(self) -> str:
        return "fake"

    def start_deadline(self) -> float:
        return time.monotonic() + 1

    def assess(self, message, recent_jobs, deadline=None):
        self.calls.append(("assess", message, [job.get("transcript") for job in recent_jobs]))
        return None if self.fail else dict(self.answers.get(message, self.QUIET))

    def is_correction(self, message, earlier_jobs, deadline=None):
        self.calls.append(("is_correction", message, [job.get("transcript") for job in earlier_jobs]))
        return None if self.fail_previous else self.previous.get(message, 0.03)

    def warm(self) -> None:
        self.calls.append(("warm",))


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

    def test_available_but_unconfigured_models_are_never_chosen_by_default(self) -> None:
        # e.g. a pay-per-use API key exposes models the user never picked for Ceviz.
        rows = [{"key": "anthropic/claude-sonnet-5", "available": True, "tags": []},
                {"key": "anthropic/claude-haiku-4-5", "available": True, "tags": ["fallback#1"]}]
        roles = rank(rows)
        self.assertEqual(roles["balanced"], [])
        self.assertEqual([m.ref for m in roles["light"]], ["anthropic/claude-haiku-4-5"])
        self.assertEqual(rank(rows, include_unconfigured=True)["balanced"][0].ref, "anthropic/claude-sonnet-5")

    def test_discovery_failure_falls_back_to_stale_cache_then_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "models.json"
            cache.write_text(json.dumps({"agent": "cevizmain", "fetched_at": 0, "models": CATALOG_ROWS}))

            def broken():
                raise RuntimeError("gateway down")

            self.assertEqual(ModelCatalog("cevizmain", cache, runner=broken).best("balanced").ref, SONNET)
            self.assertIsNone(ModelCatalog("cevizmain", Path(tmp) / "none.json", runner=broken).best("balanced"))

    def test_stale_cache_is_served_at_once_and_refreshed_in_the_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "models.json"
            cache.write_text(json.dumps({"agent": "cevizmain", "fetched_at": 0, "models": CATALOG_ROWS}))
            release = threading.Event()
            fresh_rows = [{"key": "anthropic/claude-sonnet-6", "available": True, "tags": ["configured"]}]

            def slow():
                release.wait(5)
                return fresh_rows

            catalog = ModelCatalog("cevizmain", cache, runner=slow)
            started = time.monotonic()
            self.assertEqual(catalog.best("balanced").ref, SONNET)  # stale answer, no waiting
            self.assertLess(time.monotonic() - started, 1.0)
            release.set()
            for _ in range(100):
                if catalog.best("balanced").ref == "anthropic/claude-sonnet-6":
                    break
                time.sleep(0.02)
            self.assertEqual(catalog.best("balanced").ref, "anthropic/claude-sonnet-6")
            self.assertEqual(json.loads(cache.read_text())["models"], fresh_rows)

    def test_no_cache_does_not_block_while_a_warm_up_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = threading.Event()
            calls = []

            def slow():
                calls.append(1)
                release.wait(5)
                return list(CATALOG_ROWS)

            catalog = ModelCatalog("cevizmain", Path(tmp) / "models.json", runner=slow)
            catalog.warm()
            for _ in range(100):
                if calls:
                    break
                time.sleep(0.01)
            started = time.monotonic()
            self.assertIsNone(catalog.best("balanced"))  # still warming: agent default, no wait
            self.assertLess(time.monotonic() - started, 1.0)
            release.set()
            for _ in range(100):
                if catalog.best("balanced"):
                    break
                time.sleep(0.02)
            self.assertEqual(catalog.best("balanced").ref, SONNET)
            self.assertEqual(len(calls), 1)

    def test_in_process_catalog_refreshes_after_the_ttl(self) -> None:
        rows = [list(CATALOG_ROWS)]
        calls = []

        def runner():
            calls.append(1)
            return rows[0]

        catalog = ModelCatalog("cevizmain", runner=runner)
        catalog.rows()
        self.assertEqual(len(calls), 1)
        catalog._loaded_at -= 7 * 3600
        catalog.rows()
        for _ in range(100):
            if len(calls) == 2:
                break
            time.sleep(0.01)
        self.assertEqual(len(calls), 2)


class TestPusulaRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.jev = MagicMock(spec=JevClient)
        self.jev.is_configured = True
        self.jev.evaluate_boolean.return_value = 0.1
        self.judge = FakeJudge()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _pusula(self, rows=CATALOG_ROWS) -> CevizPusula:
        return CevizPusula(state_dir=self.state_dir, jev_client=self.jev, catalog=fake_catalog(rows),
                           agent="cevizmain", judge=self.judge)

    def _flag(self, message: str, correction: float = 0.04, repeat: float = 0.06) -> None:
        self.judge.answers[message] = {"correction": correction, "repeat": repeat}

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
        self.assertEqual([call[0] for call in self.judge.calls], ["assess", "assess"])

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
        message = "Nasıl ya? Tarayıcıyı nasıl açamıyorsun?"  # pushback the old regex missed
        self._flag(message, correction=0.93)
        decision = self._pusula().route(message)
        self.assertTrue(decision.escalated)
        self.assertEqual((decision.model, decision.thinking), (SONNET, "medium"))
        self.assertEqual((decision.escalation_level, decision.escalation_signals), (1, ("correction",)))
        self.assertEqual(decision.jev_calls, 0)

    def test_previous_job_that_missed_climbs_to_high_thinking(self) -> None:
        context = {"recent_jobs": [
            self._job("Windows node'u kur", "needs_input", ago=300),
            self._job("Hayır, sen kur", "blocked", ago=120),
        ]}
        self._flag("Olmadı, tekrar dene", correction=0.97)
        decision = self._pusula().route("Olmadı, tekrar dene", context=context)
        self.assertEqual((decision.model, decision.thinking, decision.escalation_level), (SONNET, "high", 2))
        self.assertIn("repeated_miss", decision.escalation_signals)
        self.assertNotIn("is_correction", [call[0] for call in self.judge.calls])

    def test_previous_turn_that_was_a_correction_climbs_to_high_thinking(self) -> None:
        context = {"recent_jobs": [
            self._job("Windows node'u kur", "needs_input", ago=300),
            self._job("Daha önce kurmuştuk, gayet net hatırlıyorum", "needs_input", ago=120),
        ]}
        self._flag("Hâlâ aynı şeyi söylüyorsun", correction=0.91)
        self.judge.previous["Daha önce kurmuştuk, gayet net hatırlıyorum"] = 0.88
        decision = self._pusula().route("Hâlâ aynı şeyi söylüyorsun", context=context)
        self.assertEqual((decision.thinking, decision.escalation_level), ("high", 2))
        # The previous turn is judged against the turns before it, with the same question.
        self.assertIn(("is_correction", "Daha önce kurmuştuk, gayet net hatırlıyorum", ["Windows node'u kur"]),
                      self.judge.calls)

    def test_previous_turn_that_was_not_a_correction_stays_on_level_one(self) -> None:
        context = {"recent_jobs": [self._job("Zoho IMAP ayarını bul", "needs_input")]}
        self._flag("Yanlış yere baktın", correction=0.9)
        decision = self._pusula().route("Yanlış yere baktın", context=context)
        self.assertEqual((decision.thinking, decision.escalation_level), ("medium", 1))

    def test_repeating_an_unanswered_request_escalates_without_a_correction(self) -> None:
        context = {"recent_jobs": [self._job("Claude Code oturumlarına erişebiliyor musun?", "needs_input")]}
        self._flag("Claude Code oturumlarına erişebiliyor musun", repeat=0.81)
        decision = self._pusula().route("Claude Code oturumlarına erişebiliyor musun", context=context)
        self.assertEqual(decision.escalation_level, 1)
        self.assertEqual(decision.escalation_signals, ("repeat",))

    def test_a_failed_job_alone_does_not_escalate_and_old_jobs_are_not_evidence(self) -> None:
        outage = {"recent_jobs": [self._job("PR'ı kontrol et", "unknown", status="failed")]}
        self.assertFalse(self._pusula().route("Selam", context=outage).escalated)
        stale = {"recent_jobs": [self._job("Windows node'u kur", "blocked", ago=3600)]}
        self._flag("Olmadı", correction=0.95)
        self.assertEqual(self._pusula().route("Olmadı", context=stale).escalation_level, 1)
        self.assertEqual(self.judge.calls[-1], ("assess", "Olmadı", []))

    def test_threshold_comes_from_the_config(self) -> None:
        self._edit(escalation_threshold=0.95)
        self._flag("Yanlış yaptın", correction=0.93)
        self.assertFalse(self._pusula().route("Yanlış yaptın").escalated)

    def test_judge_that_does_not_answer_keeps_the_agent_default(self) -> None:
        self._with_light_tier()
        self.jev.evaluate_boolean.return_value = 0.99
        self.judge.fail = True
        decision = self._pusula().route("Selam, nasılsın?")
        self.assertEqual((decision.model, decision.reason, decision.fallback), (None, "judge_failed", True))
        self.assertFalse(decision.escalated)
        self.jev.evaluate_boolean.assert_not_called()  # no other model guesses in its place

    def test_previous_turn_check_that_fails_keeps_the_agent_default(self) -> None:
        context = {"recent_jobs": [self._job("Zoho IMAP ayarını bul", "needs_input")]}
        self._flag("Yanlış yere baktın", correction=0.9)
        self.judge.fail_previous = True
        decision = self._pusula().route("Yanlış yere baktın", context=context)
        self.assertEqual((decision.model, decision.reason), (None, "judge_failed"))

    def test_configured_ladder_overrides_discovery(self) -> None:
        self._edit(escalation_models=["anthropic/claude-sonnet-4-6", "anthropic/claude-opus-5"],
                   escalation_thinking=["low", "high"])
        pusula = self._pusula()
        self.assertEqual([(m.id, m.thinking) for m in pusula.escalation_ladder()],
                         [("anthropic/claude-sonnet-4-6", "low"), ("anthropic/claude-opus-5", "high")])

    def test_no_discoverable_model_escalates_on_the_agent_default(self) -> None:
        self._flag("Yanlış yaptın", correction=0.9)
        decision = self._pusula(rows=CATALOG_ROWS[:2]).route("Yanlış yaptın")
        self.assertTrue(decision.escalated)
        self.assertIsNone(decision.model)

    def test_single_turn_mode_ignores_context_and_correction(self) -> None:
        self._with_light_tier()
        self._edit(routing_mode=MODE_SINGLE_TURN)
        self.jev.evaluate_boolean.return_value = 0.9
        self._flag("Yanlış yaptın, bunu düzelt", correction=0.95)
        decision = self._pusula().route("Yanlış yaptın, bunu düzelt", context={"continuation": "ağır iş"})
        self.assertFalse(decision.escalated)
        self.assertFalse(decision.context_used)
        self.assertEqual(decision.model, LIGHT)
        self.assertEqual(self.judge.calls, [])

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
        self._flag("Bu tamamen yanlış", correction=0.9)
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
                decision = CevizPusula(state_dir=tmp, catalog=fake_catalog(), judge=FakeJudge()).route("Selam, nasılsın?")
        self.assertEqual((decision.reason, decision.model), ("light_turn", LIGHT))
        sent = self._sent(opener)[2]["questions"]["eval_bool"]
        self.assertEqual(sent["instructions"], "Is this only small talk or a general-knowledge question?")


class TestEscalationJudge(unittest.TestCase):
    @staticmethod
    def _reply(*tops: list) -> MagicMock:
        """urlopen stand-in whose successive answers carry these first-token top log probabilities."""
        responses = []
        for top in tops:
            response = MagicMock()
            response.read.return_value = json.dumps({"logprobs": [{"top_logprobs": top}] if top is not None else []}).encode()
            responses.append(response)
        opener = MagicMock()
        opener.return_value.__enter__.side_effect = responses
        return opener

    @staticmethod
    def _bodies(opener: MagicMock) -> list[dict]:
        return [json.loads(call.args[0].data) for call in opener.call_args_list]

    def test_reads_letters_and_yes_no_words_as_the_same_option(self) -> None:
        self.assertAlmostEqual(letter_probability([{"token": "A", "logprob": -0.1}, {"token": " B", "logprob": -2.4}]),
                               math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.4)))
        self.assertAlmostEqual(letter_probability([{"token": "Yes", "logprob": -0.2}, {"token": "[A]", "logprob": -1.0},
                                                   {"token": "no", "logprob": -3.0}]),
                               (math.exp(-0.2) + math.exp(-1.0)) / (math.exp(-0.2) + math.exp(-1.0) + math.exp(-3.0)))
        self.assertIsNone(letter_probability([{"token": "Hmm", "logprob": -0.1}]))
        self.assertIsNone(letter_probability(None))

    def test_state_holds_the_last_turns_and_comes_before_the_question(self) -> None:
        now = time.time()
        jobs = [{"transcript": f"istek {i}", "outcome": "done", "status": "completed", "created_at": now - 60 * i}
                for i in (5, 4, 3, 2)]
        state = build_state("yeni mesaj", jobs, history=3, now=now)
        self.assertEqual([r["request"] for r in state["recent_requests"]], ["istek 4", "istek 3", "istek 2"])
        self.assertEqual([r["minutes_ago"] for r in state["recent_requests"]], [4, 3, 2])
        self.assertEqual(build_state("x", jobs, history=0)["recent_requests"], [])
        prompt = build_prompt(state, QUESTIONS["correction"])
        self.assertLess(prompt.index("State:"), prompt.index("Question:"))
        self.assertTrue(prompt.endswith("Answer with the letter of the best option only."))

    def test_asks_both_questions_on_the_local_model(self) -> None:
        opener = self._reply([{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.0}],
                             [{"token": "No", "logprob": -0.02}, {"token": "Yes", "logprob": -4.0}])
        judge = EscalationJudge(url="http://127.0.0.1:11434/", model="openjev")
        with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen", opener):
            answers = judge.assess("Olmadı", [])
        self.assertGreater(answers["correction"], 0.9)
        self.assertLess(answers["repeat"], 0.1)
        self.assertEqual(opener.call_args.args[0].full_url, "http://127.0.0.1:11434/api/chat")
        body = self._bodies(opener)[0]
        self.assertEqual((body["model"], body["think"], body["logprobs"]), ("openjev", False, True))
        self.assertEqual(body["options"]["num_predict"], 1)
        self.assertEqual(body["options"]["num_ctx"], 2048)

    def test_any_failure_means_no_answer(self) -> None:
        import urllib.error
        judge = EscalationJudge()
        for failure in (urllib.error.URLError("refused"), TimeoutError(),
                        urllib.error.HTTPError("http://x", 404, "model not found", {}, io.BytesIO(b"not found"))):
            with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen", side_effect=failure):
                self.assertIsNone(judge.assess("Olmadı", []))
        with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen", self._reply(None)):
            self.assertIsNone(judge.assess("Olmadı", []))
        with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen",
                   self._reply([{"token": "Maybe", "logprob": -0.1}])):
            self.assertIsNone(judge.assess("Olmadı", []))

    def test_a_passed_deadline_sends_nothing(self) -> None:
        opener = self._reply([{"token": "A", "logprob": -0.1}])
        with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen", opener):
            self.assertIsNone(EscalationJudge().assess("Olmadı", [], deadline=time.monotonic() - 1))
        opener.assert_not_called()

    def test_model_without_a_thinking_switch_is_asked_again_without_it(self) -> None:
        import urllib.error
        rejection = urllib.error.HTTPError("http://x", 400, "Bad Request", {},
                                           io.BytesIO(b'{"error":"model does not support thinking"}'))
        answer = MagicMock()
        answer.read.return_value = json.dumps({"logprobs": [{"top_logprobs": [{"token": "B", "logprob": -0.1}]}]}).encode()
        opener = MagicMock(side_effect=[rejection, MagicMock(__enter__=MagicMock(return_value=answer)),
                                        MagicMock(__enter__=MagicMock(return_value=answer))])
        judge = EscalationJudge()
        with patch("ceviz_pusula.escalation_judge.urllib.request.urlopen", opener):
            self.assertIsNotNone(judge.assess("Olmadı", []))
        bodies = self._bodies(opener)
        self.assertEqual(["think" in body for body in bodies], [True, False, False])

    def test_config_reads_judge_settings_with_safe_defaults(self) -> None:
        defaults = PusulaConfig.from_dict({})
        self.assertEqual((defaults.escalation_judge_model, defaults.escalation_threshold, defaults.escalation_history),
                         ("hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0", 0.5, 3))
        custom = PusulaConfig.from_dict({"escalation_threshold": 2, "escalation_history": 0,
                                         "escalation_judge_timeout_ms": "nope", "escalation_judge_model": "m"})
        self.assertEqual((custom.escalation_threshold, custom.escalation_history), (1.0, 0))
        self.assertEqual((custom.escalation_judge_timeout_ms, custom.escalation_judge_model), (1200, "m"))
        self.assertEqual(PusulaConfig.from_dict(custom.to_dict()).escalation_judge_model, "m")


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
