"""The local decision model behind every Pusula decision: escalation and the light tier.

It replaced the correction regex. On the escalation-detector set (experiments/escalation-detector,
three prior turns) the regex reached precision 0.53 and recall 0.37, while APUS-OpenJev-v1 4B
reached AUC 0.997 with precision 0.92 and recall 0.99 at 0.5.

The model is served by Ollama, and each answer is read from the first token's log probabilities.
Every failure (unreachable server, missing model, deadline, an answer without the option letters)
returns None, and the caller keeps the turn on the agent default: a decision model that cannot
answer must never move a turn or fail a command.
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

logger = logging.getLogger("ceviz.pusula.decisions")

DEFAULT_DECISION_URL = "http://127.0.0.1:11434"
DEFAULT_DECISION_MODEL = "hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0"
# One budget for the two to four questions of a turn, asked one after another. A warm model answers
# each in 0.2-0.5 s, but the first turn after the model sat idle beside other GPU work measured
# 1.27 s + 0.98 s, which the earlier 1.2 s budget cut off. The plugin keeps it below the host's
# routing timeout (see plugin.decision_budget_ms).
DEFAULT_DECISION_TIMEOUT_MS = 2500
DEFAULT_ESCALATION_HISTORY = 3
DEFAULT_ESCALATION_THRESHOLD = 0.5
# The model's own default context is 9216 tokens; the prompt stays well under 2048, and the smaller
# window keeps the model beside Whisper on a 16 GB card.
CONTEXT_TOKENS = 2048
# Every routed turn uses the model and its memory is budgeted beside Whisper, so it stays loaded
# until Ollama stops. A timed unload would make the next turn pay a cold load (about 4 s), which
# does not fit the routing budget.
KEEP_ALIVE = -1
# A model that answered this recently answers the next question at full speed (measured after 30 s
# idle), so a wake-up call would only add load. After 120 s idle a wake-up made the next turn
# about a third faster.
AWAKE_SECONDS = 30
# A cold load takes about 4 s; a card busy with other work can take much longer.
LOAD_TIMEOUT_SECONDS = 120
MAX_TRANSCRIPT_CHARS = 600

# Worded as measured in the experiment; changing them invalidates its numbers.
ESCALATION_QUESTIONS = {
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

# The light-tier question is not measured yet; operators can replace the instructions.
LIGHT_OPTIONS = {
    "true": "Yes: it is small talk or a general-knowledge question that can be answered from memory alone.",
    "false": "No: it may need a tool, command, file, web lookup, live system state, earlier conversation "
             "or delegation.",
}

# Options read "[A] Yes: ..." / "[B] No: ...". APUS-OpenJev answers with the word about as often as
# with the letter, so both count for the same option.
_OPTION_TOKENS = {"A": "A", "B": "B", "YES": "A", "NO": "B"}


class DecisionUnavailable(Exception):
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


class DecisionModel:
    def __init__(
        self,
        url: str = DEFAULT_DECISION_URL,
        model: str = DEFAULT_DECISION_MODEL,
        timeout_ms: int = DEFAULT_DECISION_TIMEOUT_MS,
        history: int = DEFAULT_ESCALATION_HISTORY,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_ms = timeout_ms
        self.history = history
        self._send_think = True
        # Monotonic time of the model's last answer. None, not 0: monotonic time starts near zero
        # at boot, so 0 would read as "answered just now" on a freshly started machine.
        self._last_answer: float | None = None
        self._waking = False
        self._wake_lock = threading.Lock()

    def describe(self) -> str:
        return f"{self.model}@{self.url}"

    def start_deadline(self) -> float:
        """One budget for every question of a turn, so the turn stays inside the host's routing timeout."""
        return time.monotonic() + self.timeout_ms / 1000

    def assess(self, message: str, recent_jobs: list[dict[str, Any]], deadline: float | None = None) -> dict[str, float] | None:
        """P(yes) for each escalation question about `message`, or None if any answer is missing."""
        deadline = self.start_deadline() if deadline is None else deadline
        state = build_state(message, recent_jobs, self.history)
        answers: dict[str, float] = {}
        try:
            for name, question in ESCALATION_QUESTIONS.items():
                answers[name] = self._ask(build_prompt(state, question), deadline)
        except DecisionUnavailable as exc:
            logger.warning(f"[pusula.decisions] {exc}")
            return None
        return answers

    def is_correction(self, message: str, earlier_jobs: list[dict[str, Any]], deadline: float | None = None) -> float | None:
        """P(yes) for the correction question alone, used to check whether an earlier turn was a miss."""
        question = ESCALATION_QUESTIONS["correction"]
        return self._ask_one(build_prompt(build_state(message, earlier_jobs, self.history), question), deadline)

    def is_light(self, message: str, instructions: str, deadline: float | None = None) -> float | None:
        """P(yes) that a fresh turn needs nothing beyond the model's memory."""
        question = {"instructions": instructions, **LIGHT_OPTIONS}
        return self._ask_one(build_prompt({"new_message": message[:MAX_TRANSCRIPT_CHARS]}, question), deadline)

    def warm(self) -> None:
        """Loads the model in the background and runs it once, so the first routed turn pays for neither.

        A load alone leaves the first question to put the weights to work; one token does that here.
        """
        self._run_once_in_background("warm-up")

    def wake(self) -> bool:
        """Runs the model once in the background unless it answered in the last AWAKE_SECONDS.

        For a turn that is on its way (see PusulaRouter.prepare): the model gets its weights back
        into use while the host is still working out the words. Returns at once, never raises,
        and starts nothing while an earlier wake-up is still running.
        """
        if self._last_answer is not None and time.monotonic() - self._last_answer < AWAKE_SECONDS:
            return False
        return self._run_once_in_background("wake-up")

    def _run_once_in_background(self, purpose: str) -> bool:
        with self._wake_lock:
            if self._waking:
                return False
            self._waking = True

        def run() -> None:
            body: dict[str, Any] = {
                "model": self.model,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                # The same context window as the questions, or Ollama would reload the model.
                "options": {"temperature": 0, "num_predict": 1, "num_ctx": CONTEXT_TOKENS},
                "messages": [{"role": "user", "content": "A"}],
            }
            if self._send_think:
                body["think"] = False
            try:
                self._post("/api/chat", body, timeout=LOAD_TIMEOUT_SECONDS, retry_without_think=True)
            except Exception as exc:  # the turn itself will try again
                logger.warning(f"[pusula.decisions] {purpose} failed: {exc}")
            finally:
                with self._wake_lock:
                    self._waking = False

        threading.Thread(target=run, name=f"pusula-decisions-{purpose}", daemon=True).start()
        return True

    def _ask_one(self, prompt: str, deadline: float | None) -> float | None:
        try:
            return self._ask(prompt, self.start_deadline() if deadline is None else deadline)
        except DecisionUnavailable as exc:
            logger.warning(f"[pusula.decisions] {exc}")
            return None

    def _ask(self, prompt: str, deadline: float) -> float:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 10,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": CONTEXT_TOKENS},
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._send_think:
            body["think"] = False
        data = self._post("/api/chat", body, timeout=self._remaining(deadline), retry_without_think=True)
        tokens = data.get("logprobs") or []
        probability = letter_probability(tokens[0].get("top_logprobs")) if tokens and isinstance(tokens[0], dict) else None
        if probability is None:
            raise DecisionUnavailable("answer had no option letter in its first token")
        return probability

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            raise DecisionUnavailable("deadline passed")
        return remaining

    def _post(self, path: str, body: dict[str, Any], timeout: float, retry_without_think: bool = False) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            self._last_answer = time.monotonic()
            return data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            if retry_without_think and "think" in body and "think" in detail:
                # Models without a thinking switch reject the field; ask again without it. Only the
                # first such call pays for the round trip; later calls leave the field out.
                self._send_think = False
                retry = {k: v for k, v in body.items() if k != "think"}
                return self._post(path, retry, timeout=timeout)
            raise DecisionUnavailable(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise DecisionUnavailable(f"{type(exc).__name__}: {exc}") from exc
