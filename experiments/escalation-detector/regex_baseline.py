"""The correction regex and near-repeat check that Pusula v0.1.0 used to escalate.

Pusula now asks a decision model instead; this frozen copy keeps the regex baseline of the
experiment reproducible.
"""
import difflib
import re
import time

CORRECTION_PATTERN = re.compile(
    r"\b(anlamad[ıi]n|anlamam[ıi][sş]s[ıi]n|yanl[ıi][sş]|olmad[ıi]|tekrar dene|yeniden yap|"
    r"hata ald[ıi]m|hata verdi|ba[sş]ar[ıi]s[ıi]z|bunu demek istemedim|onu demedim|kastetmedim|"
    r"d[üu]zelt|tam tersi|eksik kald[ıi]|alakas[ıi]z|sa[cç]malad[ıi]n|[cç]al[ıi][sş]mad[ıi]|"
    r"i[sş]e yaramad[ıi]|yapamad[ıi]n|"
    r"not what i (?:meant|asked)|that'?s (?:wrong|not it)|(?:did|does)\s?n[o']?t work|try again)\b",
    re.IGNORECASE,
)
MISSED_OUTCOMES = frozenset({"blocked", "unknown"})
REPEAT_SIMILARITY = 0.8
WINDOW_SECONDS = 900


def _normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def escalation_level(prompt, recent_jobs, now=None):
    """(level, signals) exactly as Pusula v0.1.0 computed them."""
    now = time.time() if now is None else now
    recent = sorted(
        (job for job in recent_jobs if now - float(job.get("created_at") or 0) <= WINDOW_SECONDS),
        key=lambda job: float(job.get("created_at") or 0),
    )
    signals = []
    if CORRECTION_PATTERN.search(prompt):
        signals.append("correction")
    normalized = _normalize(prompt)
    if normalized and any(
        (job.get("outcome") or "") != "done"
        and difflib.SequenceMatcher(None, normalized, _normalize(job.get("transcript") or "")).ratio() >= REPEAT_SIMILARITY
        for job in recent
    ):
        signals.append("repeat")
    if not signals:
        return 0, ()
    previous = recent[-1] if recent else None
    missed = previous is not None and (
        (previous.get("outcome") or "") in MISSED_OUTCOMES or previous.get("status") == "failed"
    )
    if missed or any(CORRECTION_PATTERN.search(job.get("transcript") or "") for job in recent):
        signals.append("repeated_miss")
        return 2, tuple(signals)
    return 1, tuple(signals)
