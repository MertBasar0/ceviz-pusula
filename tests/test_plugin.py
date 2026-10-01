from __future__ import annotations

import json
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ceviz_pusula.config_guard import ConfigGuard
from ceviz_pusula.plugin import API_VERSION, PusulaRouter, create

ROOT = Path(__file__).resolve().parents[1]
SONNET = "anthropic/claude-sonnet-5"


class PluginContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Start from the default config, with an explicit escalation model so no discovery runs.
        guard = ConfigGuard(state_dir=self.tmp.name)
        guard.load_config()
        data = json.loads(guard.config_path.read_text(encoding="utf-8"))
        data["escalation_models"] = [SONNET]
        guard.config_path.write_text(json.dumps(data), encoding="utf-8")
        self.router = create({"api_version": 1, "agent": "cevizmain", "state_dir": self.tmp.name})

    def request(self, transcript: str, recent_jobs=None, continuation=None) -> dict:
        return {
            "api_version": 1,
            "agent": "cevizmain",
            "transcript": transcript,
            "locale": "tr-TR",
            "continuation": continuation,
            "recent_jobs": recent_jobs or [],
        }

    def test_declares_contract_v1(self) -> None:
        self.assertEqual(self.router.api_version, 1)
        self.assertEqual(API_VERSION, 1)
        with self.assertRaises(RuntimeError):
            PusulaRouter({"api_version": 2, "agent": "main", "state_dir": self.tmp.name})

    def test_ordinary_turns_keep_the_agent_default(self) -> None:
        self.assertIsNone(self.router.route(self.request("Selam, nasılsın?")))
        self.assertIsNone(self.router.route(self.request("Onaylıyorum", continuation="Önceki komut: node kur")))

    def test_correction_climbs_the_ladder(self) -> None:
        answer = self.router.route(self.request("Hayır bunu kastetmedim, tekrar dene"))
        self.assertEqual(answer["model"], SONNET)
        self.assertEqual(answer["thinking"], "medium")
        self.assertTrue(answer["reason"].startswith("escalation_l1"))
        self.assertIn("correction", answer["reason"])

    def test_repeated_miss_from_recent_jobs_reaches_level_two(self) -> None:
        missed = {"transcript": "Ceviz servisinin loglarını özetle", "outcome": "blocked",
                  "status": "completed", "created_at": time.time() - 120}
        answer = self.router.route(self.request("Yine olmadı, anlamadın", recent_jobs=[missed]))
        self.assertEqual((answer["model"], answer["thinking"]), (SONNET, "high"))
        self.assertTrue(answer["reason"].startswith("escalation_l2"))

    def test_answer_is_a_plain_dict_for_the_host(self) -> None:
        engine = MagicMock()
        engine.route.return_value = MagicMock(model=None, thinking=None, reason="no_light_tier", escalation_signals=())
        router = PusulaRouter({"api_version": 1, "agent": "main", "state_dir": self.tmp.name}, engine=engine)
        jobs = [{"transcript": "a", "created_at": 1.0}, {"transcript": "b", "created_at": 2.0}]
        self.assertIsNone(router.route(self.request("x", recent_jobs=jobs, continuation="c")))
        context = engine.route.call_args.kwargs["context"]
        self.assertEqual(context["recent_jobs"], jobs)
        self.assertEqual(context["recent_job"], jobs[-1])
        self.assertEqual(context["continuation"], "c")

    def test_pyproject_registers_the_ceviz_entry_point(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["entry-points"]["ceviz.routers"], {"pusula": "ceviz_pusula.plugin:create"})
        self.assertEqual(project.get("dependencies", []), [])


if __name__ == "__main__":
    unittest.main()
