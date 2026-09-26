from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("ceviz.pusula.jev")

VERCEL_AI_GATEWAY_ENDPOINT = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
DEFAULT_DECISION_MODEL = "typesafe-ai/jev"
DEFAULT_TIMEOUT_MS = 3500


def resolve_ai_gateway_key(env_path: Path | str | None = None) -> str:
    """Finds AI_GATEWAY_API_KEY from process environment or state .env file."""
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if key:
        return key

    candidate_paths = [
        Path(env_path) if env_path else None,
        Path.home() / ".openclaw" / "ceviz-state" / ".env",
        Path.home() / ".ocm" / "envs" / "nemo" / ".openclaw" / ".env",
        Path.home() / ".openclaw" / ".env",
    ]

    for p in candidate_paths:
        if p and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("AI_GATEWAY_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
            except Exception:
                continue

    return ""


class JevEvaluationError(Exception):
    pass


class JevClient:
    """Lightweight client for typesafe-ai/jev evaluation models on Vercel AI Gateway."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = VERCEL_AI_GATEWAY_ENDPOINT,
        model: str = DEFAULT_DECISION_MODEL,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self.api_key = api_key or resolve_ai_gateway_key()
        self.endpoint = endpoint
        self.model = model
        self.default_timeout_ms = default_timeout_ms

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def evaluate_boolean(
        self,
        state: Any,
        instructions: str,
        timeout_ms: int | None = None,
    ) -> float | None:
        """Evaluates a boolean question against state.

        Returns:
            probability (0.0 to 1.0) of true, or None if evaluation failed.
        """
        if not self.is_configured:
            logger.debug("[jev] API key not configured; skipping boolean evaluation")
            return None

        question_id = "eval_bool"
        questions = {
            question_id: {
                "type": "boolean",
                "instructions": instructions,
            }
        }

        resp_data = self._post_evaluation(state, questions, timeout_ms)
        if not resp_data:
            return None

        answers = resp_data.get("answers", {})
        ans = answers.get(question_id, {})
        if ans.get("type") == "boolean" and isinstance(ans.get("probability"), (int, float)):
            return float(ans["probability"])

        logger.warning(f"[jev] Unexpected boolean answer shape: {ans}")
        return None

    def evaluate_choice(
        self,
        state: Any,
        criteria: dict[str, str],
        instructions: str = "Select the option that best matches the given state.",
        timeout_ms: int | None = None,
    ) -> tuple[str, dict[str, float]] | None:
        """Evaluates a multiple choice question with criteria against state.

        Returns:
            (winning_choice, probabilities_dict) or None if evaluation failed.
        """
        if not self.is_configured:
            logger.debug("[jev] API key not configured; skipping choice evaluation")
            return None

        if not criteria:
            return None

        question_id = "eval_choice"
        questions = {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }

        resp_data = self._post_evaluation(state, questions, timeout_ms)
        if not resp_data:
            return None

        answers = resp_data.get("answers", {})
        ans = answers.get(question_id, {})
        if ans.get("type") == "choice" and isinstance(ans.get("choice"), str):
            choice = ans["choice"]
            raw_probs = ans.get("probabilities", {})
            probs = {k: float(v) for k, v in raw_probs.items() if isinstance(v, (int, float))}
            return choice, probs

        logger.warning(f"[jev] Unexpected choice answer shape: {ans}")
        return None

    def _post_evaluation(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        timeout_ms: int | None = None,
    ) -> dict[str, Any] | None:
        effective_timeout = (timeout_ms or self.default_timeout_ms) / 1000.0

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-auth-method": "api-key",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-model-id": self.model,
        }

        payload = {
            "state": state,
            "questions": questions,
            "providerOptions": {},
        }

        start_t = time.perf_counter()
        try:
            with httpx.Client(timeout=effective_timeout) as client:
                res = client.post(self.endpoint, headers=headers, json=payload)
                elapsed_ms = int((time.perf_counter() - start_t) * 1000)

                if res.status_code == 200:
                    logger.debug(f"[jev] Evaluation successful in {elapsed_ms}ms")
                    return res.json()

                logger.warning(
                    f"[jev] HTTP {res.status_code} in {elapsed_ms}ms: {res.text[:200]}"
                )
                return None

        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            logger.warning(f"[jev] Evaluation timed out after {elapsed_ms}ms (limit: {effective_timeout*1000:.0f}ms)")
            return None
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            logger.warning(f"[jev] Evaluation transport error in {elapsed_ms}ms: {e}")
            return None
