"""Metrics from confusion matrices and seed-paired statistics (standard library only).

    python -m tools.metrics            # print the summary and per-class tables
    python -m tools.metrics --write    # also rewrite results/derived/*.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"


# ----------------------------------------------------------------- loading
def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_confusion(name: str) -> dict:
    return json.loads((RESULTS / "confusion_matrices" / name).read_text(encoding="utf-8"))


def benchmark_matrices() -> dict[str, dict]:
    return load_confusion("medical_benchmarks_base.json")["datasets"]


def cross_dataset_matrix() -> dict:
    return load_confusion("cross_dataset_brain_mri_base.json")


# ----------------------------------------------------------------- metrics
def confusion_metrics(matrix: list[list[int]]) -> dict:
    """Accuracy, macro / weighted precision-recall-F1 and Cohen's kappa.

    Rows are true classes, columns predicted classes.
    """
    k = len(matrix)
    n = sum(map(sum, matrix))
    support = [sum(row) for row in matrix]
    predicted = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    tp = [matrix[i][i] for i in range(k)]
    precision = [tp[i] / predicted[i] if predicted[i] else 0.0 for i in range(k)]
    recall = [tp[i] / support[i] if support[i] else 0.0 for i in range(k)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    specificity = [(n - support[i] - predicted[i] + tp[i]) / (n - support[i]) for i in range(k)]
    po = sum(tp) / n
    pe = sum(support[i] * predicted[i] for i in range(k)) / (n * n)
    weighted = lambda values: sum(v * s for v, s in zip(values, support)) / n  # noqa: E731
    return {
        "n": n,
        "classes": k,
        "accuracy": po,
        "macro_precision": sum(precision) / k,
        "macro_recall": sum(recall) / k,
        "macro_f1": sum(f1) / k,
        "weighted_precision": weighted(precision),
        "weighted_recall": weighted(recall),
        "weighted_f1": weighted(f1),
        "macro_specificity": sum(specificity) / k,
        "kappa": (po - pe) / (1 - pe),
        "per_class": [{"precision": p, "recall": r, "f1": f, "support": s}
                      for p, r, f, s in zip(precision, recall, f1, support)],
    }


def wilcoxon_signed_rank_exact(x: list[float], y: list[float]) -> dict:
    """Exact Wilcoxon signed-rank test for small paired samples (no zero differences)."""
    d = [a - b for a, b in zip(x, y)]
    if any(v == 0 for v in d):
        raise ValueError("zero differences are not supported by the exact test here")
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    ranks = [0.0] * len(d)
    i = 0
    while i < len(order):                       # average ranks for ties in |d|
        j = i
        while j + 1 < len(order) and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    w_minus = sum(r for r, v in zip(ranks, d) if v < 0)
    total = 0
    at_least = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        if sum(r for r, s in zip(ranks, signs) if s > 0) >= w_plus:
            at_least += 1
    p_one = at_least / total
    return {
        "n": len(d),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_one_sided": p_one,
        "p_two_sided": min(1.0, 2 * min(p_one, 1 - p_one + 1 / total)),
        "rank_biserial": (w_plus - w_minus) / (w_plus + w_minus),
        "mean_difference": statistics.mean(d),
    }


def seed_summary() -> dict:
    rows = read_csv("seeds_pathmnist.csv")
    out = {}
    for metric in ("oa", "f1", "precision", "recall", "auc"):
        a = [float(r[metric]) for r in rows if r["model"] == "HyMaC-Net-B"]
        b = [float(r[metric]) for r in rows if r["model"] == "ResNet-50"]
        out[metric] = {
            "hymac_mean": statistics.mean(a), "hymac_sd": statistics.stdev(a),
            "hymac_min": min(a), "hymac_max": max(a),
            "resnet_mean": statistics.mean(b), "resnet_sd": statistics.stdev(b),
            "resnet_min": min(b), "resnet_max": max(b),
            **wilcoxon_signed_rank_exact(a, b),
        }
    return out


# ----------------------------------------------------------------- output
SUMMARY_FIELDS = ["dataset", "test_images", "classes", "accuracy", "macro_precision", "macro_recall",
                  "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "kappa"]


def summary_rows() -> list[dict]:
    rows = []
    for name, data in benchmark_matrices().items():
        m = confusion_metrics(data["matrix"])
        rows.append({"dataset": name, "test_images": m["n"], "classes": m["classes"],
                     **{k: round(m[k], 4) for k in SUMMARY_FIELDS[3:]}})
    cross = cross_dataset_matrix()
    m = confusion_metrics(cross["matrix"])
    rows.append({"dataset": "Cross-dataset: BRISC-25 -> Brain Tumor MRI", "test_images": m["n"],
                 "classes": m["classes"], **{k: round(m[k], 4) for k in SUMMARY_FIELDS[3:]}})
    return rows


def per_class_rows() -> list[dict]:
    rows = []
    sets = dict(benchmark_matrices())
    sets["Cross-dataset (Brain MRI)"] = cross_dataset_matrix()
    for name, data in sets.items():
        m = confusion_metrics(data["matrix"])
        for label, pc in zip(data["labels"], m["per_class"]):
            rows.append({"dataset": name, "class": label, "support": pc["support"],
                         "precision": round(pc["precision"], 4), "recall": round(pc["recall"], 4),
                         "f1": round(pc["f1"], 4)})
    return rows


def write_derived() -> None:
    DERIVED.mkdir(exist_ok=True)
    for fname, rows in (("metrics_from_confusion_matrices.csv", summary_rows()),
                        ("per_class_metrics.csv", per_class_rows())):
        with open(DERIVED / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    stats = seed_summary()
    with open(DERIVED / "seed_statistics_pathmnist.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["metric", "hymac_mean", "hymac_sd", "resnet_mean", "resnet_sd", "mean_difference",
                  "w_plus", "w_minus", "p_one_sided", "p_two_sided", "rank_biserial"]
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for metric, s in stats.items():
            w.writerow({"metric": metric, **{f: (round(s[f], 5) if isinstance(s[f], float) else s[f])
                                             for f in fields[1:]}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite results/derived/*.csv")
    args = parser.parse_args()
    print(f"{'Dataset':44}{'n':>7}{'Acc %':>8}{'macro-F1':>10}{'w-F1':>8}{'kappa':>8}")
    for r in summary_rows():
        print(f"{r['dataset']:44}{r['test_images']:>7}{100 * r['accuracy']:>8.2f}"
              f"{r['macro_f1']:>10.4f}{r['weighted_f1']:>8.4f}{r['kappa']:>8.4f}")
    print("\nPathMNIST, 5 seeds, HyMaC-Net-B vs ResNet-50 (exact Wilcoxon signed-rank):")
    for metric, s in seed_summary().items():
        print(f"  {metric:9} {s['hymac_mean']:.4f} ± {s['hymac_sd']:.4f}  vs  {s['resnet_mean']:.4f} ± "
              f"{s['resnet_sd']:.4f}   Δ {s['mean_difference']:+.4f}   p(one-sided) {s['p_one_sided']:.4f}  "
              f"p(two-sided) {s['p_two_sided']:.4f}   r = {s['rank_biserial']:.1f}")
    if args.write:
        write_derived()
        print(f"\nwrote {DERIVED.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
