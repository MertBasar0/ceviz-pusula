"""Summarize results/*.jsonl into summary.json and summary.md."""
import json
import sys
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
THRESHOLDS = (0.35, 0.5, 0.65)
ORDER = ["regex", "systemone_http_127.0.0.1_8009", "ollama_gemma4_e2b", "ollama_gemma4_e4b",
         "ollama_qwen3_14b", "ollama_qwen3_14b_qfirst"]


def load(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])["meta"], [json.loads(line) for line in lines[1:]]


def escalate_p(row):
    values = [row.get("p_correction"), row.get("p_repeat")]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def auc(rows):
    pos = [escalate_p(r) for r in rows if r["label_correction"] or r["label_repeat"]]
    neg = [escalate_p(r) for r in rows if not (r["label_correction"] or r["label_repeat"])]
    pos, neg = [p for p in pos if p is not None], [n for n in neg if n is not None]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def confusion(rows, threshold):
    tp = fp = fn = tn = missing = 0
    for row in rows:
        p = escalate_p(row)
        truth = row["label_correction"] or row["label_repeat"]
        if p is None:
            missing += 1
            continue
        predicted = p >= threshold
        tp += predicted and truth
        fp += predicted and not truth
        fn += (not predicted) and truth
        tn += (not predicted) and not truth
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "missing": missing,
            "precision": None if precision is None else round(precision, 3),
            "recall": None if recall is None else round(recall, 3),
            "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 3)}


def per_category(rows, threshold):
    out = {}
    for row in rows:
        p = escalate_p(row)
        truth = row["label_correction"] or row["label_repeat"]
        bucket = out.setdefault(row["category"], [0, 0])
        bucket[1] += 1
        bucket[0] += p is not None and ((p >= threshold) == truth)
    return {k: f"{v[0]}/{v[1]}" for k, v in out.items()}


def percentile(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 1) if values else None


def main():
    summary = {}
    files = {p.stem: p for p in RESULTS.glob("*.jsonl")}
    for name in [n for n in ORDER if n in files] + sorted(set(files) - set(ORDER)):
        meta, rows = load(files[name])
        latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        summary[name] = {
            "meta": meta,
            "errors": sum(1 for r in rows if r["error"]),
            "auc": None if name == "regex" else auc(rows),
            "at": {str(t): confusion(rows, t) for t in ((0.5,) if name == "regex" else THRESHOLDS)},
            "categories_at_0.5": per_category(rows, 0.5),
            "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        }
    if "ollama_qwen3_14b" in files and "ollama_qwen3_14b_qfirst" in files:
        a = {r["id"]: r for r in load(files["ollama_qwen3_14b"])[1]}
        b = {r["id"]: r for r in load(files["ollama_qwen3_14b_qfirst"])[1]}
        drifts, flips = [], 0
        for key in a:
            pa, pb = escalate_p(a[key]), escalate_p(b.get(key, {}))
            if pa is None or pb is None:
                continue
            drifts.append(abs(pa - pb))
            flips += (pa >= 0.5) != (pb >= 0.5)
        summary["order_check_qwen3_14b"] = {"max_drift": round(max(drifts), 3), "mean_drift": round(statistics.mean(drifts), 3),
                                            "decision_flips_at_0.5": flips, "cases": len(drifts)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["| backend | AUC | thr | precision | recall | FP | FN | p50 ms | p95 ms | VRAM/size GB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, item in summary.items():
        if name.startswith("order_check"):
            continue
        res = item["meta"].get("residency") or {}
        vram = f"{res.get('vram_gb')}/{res.get('size_gb')}" if res else "-"
        for t, c in item["at"].items():
            lines.append(f"| {name} | {item['auc'] or '-'} | {t} | {c['precision']} | {c['recall']} | {c['fp']} | {c['fn']} | "
                         f"{item['latency_ms']['p50']} | {item['latency_ms']['p95']} | {vram} |")
    lines.append("")
    cats = sorted({c for item in summary.values() if "categories_at_0.5" in item for c in item["categories_at_0.5"]})
    lines.append("| category | " + " | ".join(n for n in summary if not n.startswith("order_check")) + " |")
    lines.append("|---|" + "---|" * len([n for n in summary if not n.startswith("order_check")]))
    for cat in cats:
        lines.append(f"| {cat} | " + " | ".join(summary[n]["categories_at_0.5"].get(cat, "-")
                                               for n in summary if not n.startswith("order_check")) + " |")
    if "order_check_qwen3_14b" in summary:
        lines += ["", f"order check (qwen3:14b state-first vs question-first): {summary['order_check_qwen3_14b']}"]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
