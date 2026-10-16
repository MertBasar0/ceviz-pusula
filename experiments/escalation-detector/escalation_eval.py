"""Compare Pusula's regex escalation with decision models on the synthetic dataset.

Backends:
  regex                 Pusula v0.1.0 escalation_level, frozen in regex_baseline.py (phrases + 0.8 repeat)
  systemone:<url>       a System One decision server such as Kev (/v1/systemone, noul questions)
  ollama:<model>        a general model through Ollama, A/B letter probability from logprobs
  ollama:<model>:qfirst the same with the question before the state (prompt-order check)

Only the local machine is called. Results go to results/<backend>.jsonl.
"""
import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = {
    "correction": {
        "instructions": "Does the new message tell the assistant that its previous attempt was wrong, incomplete, "
                        "misunderstood the request, or did not work?",
        "criteria": {
            "true": "Yes: the user is pushing back on or correcting the previous attempt.",
            "false": "No: it is a new request, a follow-up, thanks, a question or reassurance, and does not say "
                     "the previous attempt failed.",
        },
    },
    "repeat": {
        "instructions": "Does the new message ask again for something a recent request already asked for, "
                        "where that recent request did not end with outcome done?",
        "criteria": {
            "true": "Yes: it re-asks a recent request whose outcome was not done.",
            "false": "No: it asks for something new, or the earlier request ended with outcome done.",
        },
    },
}


HISTORY = 5


def recent(case):
    return case["previous"][-HISTORY:]


def state_of(case):
    return {
        "recent_requests": [
            {"request": job["transcript"], "outcome": job["outcome"], "status": job["status"],
             "minutes_ago": job["minutes_ago"]}
            for job in recent(case)
        ],
        "new_message": case["message"],
    }


def post(url, body, timeout=120):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


# --- regex baseline -------------------------------------------------------------------------
def regex_backend():
    from regex_baseline import escalation_level  # Pusula v0.1.0's regex, frozen next to this script

    def ask(case):
        now = time.time()
        jobs = [
            {"transcript": job["transcript"], "outcome": job["outcome"], "status": job["status"],
             "created_at": now - job["minutes_ago"] * 60}
            for job in recent(case)
        ]
        start = time.perf_counter()
        level, signals = escalation_level(case["message"], jobs, now=now)
        elapsed = (time.perf_counter() - start) * 1000
        return {"correction": float("correction" in signals), "repeat": float("repeat" in signals),
                "level": level, "signals": list(signals)}, elapsed

    return ask, None


# --- System One (Kev) -----------------------------------------------------------------------
def systemone_backend(origin):
    endpoint = origin.rstrip("/") + "/v1/systemone"
    questions = {qid: {"type": "noul", **q} for qid, q in QUESTIONS.items()}
    # Each server names its own model; send the first one it lists.
    with urllib.request.urlopen(origin.rstrip("/") + "/v1/models", timeout=30) as response:
        listing = json.loads(response.read().decode())
    entries = listing.get("models") or listing.get("data") or []
    model_name = (entries[0].get("name") or entries[0].get("id")) if entries else None

    def ask(case):
        start = time.perf_counter()
        body = {"state": state_of(case), "questions": questions, **({"model": model_name} if model_name else {})}
        answers = post(endpoint, body).get("answers", {})
        elapsed = (time.perf_counter() - start) * 1000
        out = {}
        for qid in QUESTIONS:
            value = (answers.get(qid) or {}).get("noul")
            out[qid] = float(value) if isinstance(value, (int, float)) else None
        return out, elapsed

    return ask, None


# --- Ollama general models ------------------------------------------------------------------
OLLAMA = "http://127.0.0.1:11434"


def letter_probability(top):
    # Options read "[A] Yes: ..." and "[B] No: ...", so a model that answers with the word
    # instead of the letter (APUS-OpenJev does) is counted for the same option. Added for the
    # third round; the qwen3:14b runs were scored on letters only.
    weights = {"A": 0.0, "B": 0.0}
    words = {"YES": "A", "NO": "B"}
    for entry in top or []:
        token = re.sub(r"[\s\[\]().:*]", "", str(entry.get("token", ""))).upper()
        token = words.get(token, token)
        if token in weights:
            weights[token] += math.exp(float(entry.get("logprob", -1e9)))
    total = weights["A"] + weights["B"]
    return weights["A"] / total if total > 0 else None


def ollama_backend(model, question_first=False):
    think = {"think": False}

    def prompt(case, question):
        state = "State:\n" + json.dumps(state_of(case), ensure_ascii=False, indent=2)
        ask_text = (f"Question: {question['instructions']}\nOptions:\n"
                    f"[A] {question['criteria']['true']}\n[B] {question['criteria']['false']}")
        parts = [ask_text, state] if question_first else [state, ask_text]
        return "\n\n".join(parts) + "\n\nAnswer with the letter of the best option only."

    def call(content):
        body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 10, "keep_alive": "15m",
                "options": {"temperature": 0, "num_predict": 1},
                "messages": [{"role": "user", "content": content}], **think}
        try:
            data = post(f"{OLLAMA}/api/chat", body)
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            if "think" in detail and think:
                think.clear()
                return call(content)
            raise RuntimeError(detail[:200])
        tokens = data.get("logprobs") or []
        return letter_probability(tokens[0].get("top_logprobs")) if tokens else None

    def ask(case):
        start = time.perf_counter()
        out = {qid: call(prompt(case, q)) for qid, q in QUESTIONS.items()}
        return out, (time.perf_counter() - start) * 1000

    def residency():
        try:
            with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as response:
                for item in json.loads(response.read().decode()).get("models", []):
                    if item.get("name") == model or item.get("model") == model:
                        return {"size_gb": round(item["size"] / 1e9, 1), "vram_gb": round(item.get("size_vram", 0) / 1e9, 1)}
        except Exception:
            return None

    return ask, residency


def unload(model):
    try:
        post(f"{OLLAMA}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history", type=int, default=5)
    args = parser.parse_args()
    global HISTORY
    HISTORY = args.history
    cases = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    kind, _, rest = args.backend.partition(":")
    model = None
    if kind == "regex":
        ask, residency = regex_backend()
    elif kind == "systemone":
        ask, residency = systemone_backend(rest)
    elif kind == "ollama":
        model, _, variant = rest.partition(":qfirst") if ":qfirst" in rest else (rest, "", "")
        ask, residency = ollama_backend(model, question_first=":qfirst" in rest)
    else:
        raise SystemExit(f"unknown backend {args.backend}")

    for case in cases[:2]:  # warm-up, not recorded
        ask(case)
    meta = {"backend": args.backend, "history": HISTORY, "residency": residency() if residency else None, "cases": len(cases)}
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.backend)
    out_dir = HERE / "results" / f"h{HISTORY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
        out.write(json.dumps({"meta": meta}) + "\n")
        for index, case in enumerate(cases, 1):
            try:
                answer, elapsed = ask(case)
                error = None
            except Exception as exc:  # keep going; a failed call is recorded, never guessed
                answer, elapsed, error = {"correction": None, "repeat": None}, None, f"{type(exc).__name__}: {exc}"
            out.write(json.dumps({"id": case["id"], "category": case["category"], "lang": case["lang"],
                                  "label_correction": case["correction"], "label_repeat": case["repeat"],
                                  **{f"p_{k}": v for k, v in answer.items() if k in QUESTIONS},
                                  "extra": {k: v for k, v in answer.items() if k not in QUESTIONS},
                                  "latency_ms": None if elapsed is None else round(elapsed, 1),
                                  "error": error}, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"{args.backend}: {index}/{len(cases)}", flush=True)
    if model:
        unload(model)
    print(f"{args.backend}: done, residency={meta['residency']}", flush=True)


if __name__ == "__main__":
    main()
