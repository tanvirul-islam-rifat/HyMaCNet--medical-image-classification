"""Consistency checks for the released results (no model code, no GPU, standard library only).

    python -m tools.check_results

Every accuracy in results/ must agree with the confusion matrices or with the
support-weighted recall of the same row (weighted recall equals accuracy by
definition); every F1 must be attainable from its precision and recall; the
seed statistics and the Wilcoxon test must reproduce from the raw seed runs;
and results/derived/ must be exactly what tools/metrics.py generates.
"""
from __future__ import annotations

import csv
import io
import sys

from tools import metrics

# Official MedMNIST v2 test-split sizes for the datasets whose released matrix covers the full split.
OFFICIAL_TEST_SIZE = {"OrganAMNIST": 17778, "OCTMNIST": 1000, "PneumoniaMNIST": 624, "DermaMNIST": 2005,
                      "BreastMNIST": 156, "BloodMNIST": 3421, "TissueMNIST": 47280}
GMAC = {"HyMaC-Net-S": 0.63, "HyMaC-Net-B": 1.18}          # internal profiling, thesis Sec. 5.5.1


class Checks:
    def __init__(self, quiet: bool = False):
        self.rows: list[tuple[bool, str, str]] = []
        self.quiet = quiet

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        if not self.quiet:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<74} {detail}")
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, *_ in self.rows if not ok)


def per_dataset(variant: str) -> list[dict]:
    return metrics.read_csv(f"per_dataset_{variant}.csv")


def run(c: Checks, base: list[dict] | None = None, small: list[dict] | None = None,
        seeds_override: dict | None = None) -> Checks:
    base = base if base is not None else per_dataset("base")
    small = small if small is not None else per_dataset("small")
    tables = {"HyMaC-Net-B": base, "HyMaC-Net-S": small}

    for name, rows in tables.items():
        bad = [r["dataset"] for r in rows if abs(float(r["oa"]) / 100 - float(r["recall"])) > 5e-5]
        c.check(not bad, f"{name}: accuracy equals weighted recall in every row",
                f"{len(rows)} datasets" if not bad else ", ".join(bad))
        bad = [r["dataset"] for r in rows
               if float(r["f1"]) > (float(r["precision"]) + float(r["recall"])) / 2 + 5e-5]
        c.check(not bad, f"{name}: every F1 is attainable from its precision and recall",
                "F1 <= (P + R) / 2" if not bad else ", ".join(bad))
        fields = ("auc", "precision", "recall", "specificity", "f1", "kappa")
        bad = [r["dataset"] for r in rows if not all(0 <= float(r[f]) <= 1 for f in fields)
               or not 0 <= float(r["oa"]) <= 100]
        c.check(not bad, f"{name}: all metrics within their valid ranges", "ok" if not bad else ", ".join(bad))

    cms = metrics.benchmark_matrices()
    disagree = []
    checked = 0
    for r in base:
        if "Figure 5.1" not in r["oa_source"]:
            continue
        checked += 1
        m = metrics.confusion_metrics(cms[r["dataset"]]["matrix"])
        if abs(100 * m["accuracy"] - float(r["oa"])) > 0.005:
            disagree.append(f"{r['dataset']} {100 * m['accuracy']:.2f}")
    c.check(not disagree and checked >= 9,
            "HyMaC-Net-B accuracy matches the released confusion matrices",
            f"{checked} datasets" if not disagree else ", ".join(disagree))

    kappa_match = [r["dataset"] for r in base
                   if "Figure 5.1" in r["oa_source"]
                   and abs(metrics.confusion_metrics(cms[r["dataset"]]["matrix"])["kappa"] - float(r["kappa"])) > 5e-5]
    c.check(not kappa_match, "HyMaC-Net-B Cohen's kappa matches the released confusion matrices",
            "ok" if not kappa_match else ", ".join(kappa_match))

    sizes = [f"{k} {metrics.confusion_metrics(cms[k]['matrix'])['n']} != {v}"
             for k, v in OFFICIAL_TEST_SIZE.items() if metrics.confusion_metrics(cms[k]["matrix"])["n"] != v]
    c.check(not sizes, "Confusion matrices cover the official MedMNIST v2 test splits",
            f"{len(OFFICIAL_TEST_SIZE)} datasets" if not sizes else "; ".join(sizes))

    cross = metrics.confusion_metrics(metrics.cross_dataset_matrix()["matrix"])
    row = next(r for r in metrics.read_csv("cross_dataset.csv") if r["model"] == "HyMaC-Net-B")
    c.check(abs(100 * cross["accuracy"] - float(row["oa"])) < 0.005 and abs(cross["macro_f1"] - float(row["f1"])) < 5e-4,
            "Cross-dataset accuracy and macro-F1 match the released confusion matrix",
            f"{100 * cross['accuracy']:.2f} % on {cross['n']} test images")

    stats = seeds_override or metrics.seed_summary()
    expected = {"oa": (0.9286, 0.0106, 0.8996, 0.0085), "f1": (0.9259, 0.0113, 0.8979, 0.0080),
                "auc": (0.9886, 0.0024, 0.9740, 0.0081)}
    bad = [m for m, (a, sa, b, sb) in expected.items()
           if any(abs(x - y) > 6e-5 for x, y in zip((stats[m]["hymac_mean"], stats[m]["hymac_sd"],
                                                     stats[m]["resnet_mean"], stats[m]["resnet_sd"]),
                                                    (a, sa, b, sb)))]
    c.check(not bad, "Seed means and SDs reproduce the thesis statistics (Table 5.7)",
            "accuracy, F1, AUC" if not bad else ", ".join(bad))
    c.check(all(abs(s["p_one_sided"] - 1 / 32) < 1e-12 and s["rank_biserial"] == 1.0 for s in stats.values()),
            "Wilcoxon: all 5 seeds favour HyMaC-Net-B, exact one-sided p = 1/32",
            "two-sided p = 0.0625")

    eff = {r["model"]: r for r in metrics.read_csv("efficiency.csv")}
    c.check(all(abs(float(eff[m]["gflops"]) - 2 * g) < 1e-9 for m, g in GMAC.items()),
            "HyMaC-Net GFLOPs = 2 x profiled GMACs", "1.26 / 2.36")

    stale = []
    for fname, rows in (("metrics_from_confusion_matrices.csv", metrics.summary_rows()),
                        ("per_class_metrics.csv", metrics.per_class_rows())):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        path = metrics.DERIVED / fname
        if not path.exists() or path.read_text(encoding="utf-8") != buf.getvalue():
            stale.append(fname)
    c.check(not stale, "results/derived/ is up to date with tools/metrics.py",
            "ok" if not stale else ", ".join(stale) + " — run python -m tools.metrics --write")
    return c


def main() -> int:
    c = run(Checks())
    print(f"\n{len(c.rows) - c.failed} passed, {c.failed} failed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
