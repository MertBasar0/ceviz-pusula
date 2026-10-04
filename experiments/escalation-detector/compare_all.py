"""Per model and history: AUC, the 0.5 point, the best threshold, and the best point with at most 5 false escalations."""
from pathlib import Path

from summarize import auc, confusion, escalate_p, load, per_category, percentile

HERE = Path(__file__).resolve().parent
NAMES = {"systemone_http_127.0.0.1_8009": "Kev 0.8B", "systemone_http_127.0.0.1_8010": "Decider 2B",
         "systemone_http_127.0.0.1_8011": "this-that 1.2", "ollama_qwen3_14b": "qwen3:14b", "regex": "regex"}
GRID = [t / 100 for t in range(5, 96)]


def score(c):
    return c["tp"] - c["fp"] - c["fn"]


for h in (1, 3, 5):
    print(f"\n## history {h}")
    print("| model | AUC | @0.5 P/R/FP/FN | best thr | P | R | FP | FN | best with FP<=5: thr P R FN | p50/p95 ms |")
    for stem, label in NAMES.items():
        path = HERE / "results" / f"h{h}" / f"{stem}.jsonl"
        if not path.exists():
            continue
        rows = load(path)[1]
        lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        half = confusion(rows, 0.5)
        if stem == "regex":
            print(f"| {label} | - | {half['precision']}/{half['recall']}/{half['fp']}/{half['fn']} | - | | | | | | ~0 |")
            continue
        sweep = [(t, confusion(rows, t)) for t in GRID]
        bt, bc = max(sweep, key=lambda item: score(item[1]))
        limited = [item for item in sweep if item[1]["fp"] <= 5]
        lt, lc = max(limited, key=lambda item: score(item[1]))
        print(f"| {label} | {auc(rows)} | {half['precision']}/{half['recall']}/{half['fp']}/{half['fn']} | {bt:.2f} | "
              f"{bc['precision']} | {bc['recall']} | {bc['fp']} | {bc['fn']} | {lt:.2f} {lc['precision']} {lc['recall']} {lc['fn']} | "
              f"{percentile(lat, 0.5)}/{percentile(lat, 0.95)} |")
        if h == 3:
            print("     categories at best thr:", per_category(rows, bt))
