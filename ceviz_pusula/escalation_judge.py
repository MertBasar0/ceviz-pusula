"""Asks a local decision model whether the conversation shows a miss.

This replaced the correction regex. On the escalation-detector set (experiments/escalation-detector,
three prior turns) the regex reached precision 0.53 and recall 0.37, while APUS-OpenJev-v1 4B
reached AUC 0.997 with precision 0.92 and recall 0.99 at 0.5.

The judge talks to Ollama, which serves the GGUF build, and reads the answer from the first token's
log probabilities. Every failure (unreachable server, missing model, deadline, an answer without
the option letters) returns None, and the caller keeps the turn on the agent default: a judge that
cannot answer must never escalate a turn or fail a command.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("ceviz.pusula.judge")

DEFAULT_JUDGE_URL = "http://127.0.0.1:11434"
DEFAULT_JUDGE_MODEL = "hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0"
DEFAULT_JUDGE_TIMEOUT_MS = 1200
DEFAULT_JUDGE_HISTORY = 3
DEFAULT_ESCALATION_THRESHOLD = 0.5
# The model's own default context is 9216 tokens; the prompt stays well under 2048, and the smaller
# window keeps the model beside Whisper on a 16 GB card.
JUDGE_CONTEXT_TOKENS = 2048
JUDGE_KEEP_ALIVE = "24h"
MAX_TRANSCRIPT_CHARS = 600

# Worded as measured in the experiment; changing them invalidates its numbers.
QUESTIONS = {
    "correction": {
        "instructions": "Does the new message tell the assistant that its previous attempt was wrong, incomplete, "
                        "misunderstood the request, or did not work?",
        "true": "Yes: the user is pushing back on or correcting the previous attempt.",
        "false": "No: it is a new request, a follow-up, thanks, a question or reassurance, and does not say "
                 "the previous attempt failed.",
    },
    "repeat": {
        "instructions": "Does the new message ask again for something a recent request already asked for, "
                        "where that recent request did not end with outcome done?",
        "true": "Yes: it re-asks a recent request whose outcome was not done.",
        "false": "No: it asks for something new, or the earlier request ended with outcome done.",
    },
}

# Options read "[A] Yes: ..." / "[B] No: ...". APUS-OpenJev answers with the word about as often as
# with the letter, so both count for the same option.
_OPTION_TOKENS = {"A": "A", "B": "B", "YES": "A", "NO": "B"}


class JudgeUnavailable(Exception):
    pass


def letter_probability(top: list[dict[str, Any]] | None) -> float | None:
    """P(option A) from a first token's top log probabilities, or None when neither option shows up."""
    weights = {"A": 0.0, "B": 0.0}
    for entry in top or []:
        token = re.sub(r"[\s\[\]().:*]", "", str(entry.get("token", ""))).upper()
        option = _OPTION_TOKENS.get(token)
        if option:
            try:
                weights[option] += math.exp(float(entry.get("logprob")))
            except (TypeError, ValueError, OverflowError):
                continue
    total = weights["A"] + weights["B"]
    return weights["A"] / total if total > 0 else None


def build_state(message: str, recent_jobs: list[dict[str, Any]], history: int, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    return {
        "recent_requests": [
            {
                "request": str(job.get("transcript") or "")[:MAX_TRANSCRIPT_CHARS],
                "outcome": job.get("outcome"),
                "status": job.get("status"),
                "minutes_ago": max(0, round((now - float(job.get("created_at") or now)) / 60)),
            }
            for job in recent_jobs[-history:] if history > 0
        ],
        "new_message": message[:MAX_TRANSCRIPT_CHARS],
    }


def build_prompt(state: dict[str, Any], question: dict[str, str]) -> str:
    # State first, then the question: the experiment measured 29 false positives with the
    # question first against 3 with the state first.
    return (
        "State:\n" + json.dumps(state, ensure_ascii=False, indent=2)
        + f"\n\nQuestion: {question['instructions']}\nOptions:\n[A] {question['true']}\n[B] {question['false']}"
        + "\n\nAnswer with the letter of the best option only."
    )


class EscalationJudge:
    def __init__(
        self,
        url: str = DEFAULT_JUDGE_URL,
        model: str = DEFAULT_JUDGE_MODEL,
        timeout_ms: int = DEFAULT_JUDGE_TIMEOUT_MS,
        history: int = DEFAULT_JUDGE_HISTORY,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_ms = timeout_ms
        self.history = history
        self._send_think = True

    def describe(self) -> str:
        return f"{self.model}@{self.url}"

    def start_deadline(self) -> float:
        """One budget for every question of a turn, so the turn stays inside the host's routing timeout."""
        return time.monotonic() + self.timeout_ms / 1000

    def assess(self, message: str, recent_jobs: list[dict[str, Any]], deadline: float | None = None) -> dict[str, float] | None:
        """P(yes) for each question about `message`, or None if any answer is missing."""
        deadline = self.start_deadline() if deadline is None else deadline
        state = build_state(message, recent_jobs, self.history)
        answers: dict[str, float] = {}
        try:
            for name, question in QUESTIONS.items():
                answers[name] = self._ask(build_prompt(state, question), deadline)
        except JudgeUnavailable as exc:
            logger.warning(f"[pusula.judge] {exc}")
            return None
        return answers

    def is_correction(self, message: str, earlier_jobs: list[dict[str, Any]], deadline: float | None = None) -> float | None:
        """P(yes) for the correction question alone, used to check whether an earlier turn was a miss."""
        deadline = self.start_deadline() if deadline is None else deadline
        try:
            return self._ask(build_prompt(build_state(message, earlier_jobs, self.history), QUESTIONS["correction"]), deadline)
        except JudgeUnavailable as exc:
            logger.warning(f"[pusula.judge] {exc}")
            return None

    def warm(self) -> None:
        """Loads the model in the background so the first routed turn does not pay for it."""
        def load() -> None:
            body = {"model": self.model, "keep_alive": JUDGE_KEEP_ALIVE, "options": {"num_ctx": JUDGE_CONTEXT_TOKENS}}
            try:
                self._post("/api/generate", body, timeout=120)
            except Exception as exc:  # the first turn will try again
                logger.warning(f"[pusula.judge] warm-up failed: {exc}")

        threading.Thread(target=load, name="pusula-judge-warm", daemon=True).start()

    def _ask(self, prompt: str, deadline: float) -> float:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 10,
            "keep_alive": JUDGE_KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": JUDGE_CONTEXT_TOKENS},
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._send_think:
            body["think"] = False
        data = self._post("/api/chat", body, timeout=self._remaining(deadline), retry_without_think=True)
        tokens = data.get("logprobs") or []
        probability = letter_probability(tokens[0].get("top_logprobs")) if tokens and isinstance(tokens[0], dict) else None
        if probability is None:
            raise JudgeUnavailable("answer had no option letter in its first token")
        return probability

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            raise JudgeUnavailable("deadline passed")
        return remaining

    def _post(self, path: str, body: dict[str, Any], timeout: float, retry_without_think: bool = False) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            if retry_without_think and "think" in body and "think" in detail:
                # Models without a thinking switch reject the field; ask again without it. Only the
                # first such call pays for the round trip; later calls leave the field out.
                self._send_think = False
                retry = {k: v for k, v in body.items() if k != "think"}
                return self._post(path, retry, timeout=timeout)
            raise JudgeUnavailable(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise JudgeUnavailable(f"{type(exc).__name__}: {exc}") from exc
