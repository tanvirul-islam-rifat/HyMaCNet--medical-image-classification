"""Draw the README figures from results/ (no model code involved).

    pip install -r requirements.txt
    python -m tools.make_figures
"""
from __future__ import annotations

import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools import metrics  # noqa: E402

OUT = metrics.ROOT / "docs" / "images"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SMALL, BASE = "#2a78d6", "#eb6834"                 # categorical slots 1-2 (validated palette)
THIRD = "#1baf7a"                                  # slot 3
NEUTRAL = "#a9a8a2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.7,
})

MEDMNIST9 = ["PathMNIST", "DermaMNIST", "OCTMNIST", "PneumoniaMNIST", "BreastMNIST", "BloodMNIST",
             "OrganAMNIST", "OrganCMNIST", "OrganSMNIST"]
ORDER = ["BloodMNIST", "CPN X-ray", "OrganAMNIST", "PathMNIST", "OrganCMNIST", "OCTMNIST",
         "PneumoniaMNIST", "BreastMNIST", "Kvasir", "OrganSMNIST", "DermaMNIST", "TissueMNIST"]
COMPACT = {"ResNet-18", "MedViT-T", "MedViT-S", "MedMamba-T", "MedMamba-S", "ResNet-50"}


def ours(variant: str) -> dict[str, float]:
    return {r["dataset"]: float(r["oa"]) for r in metrics.read_csv(f"per_dataset_{variant}.csv")}


def baseline_oa() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in metrics.read_csv("baselines_medmnist.csv"):
        out.setdefault(r["model"], {})[r["dataset"]] = 100 * float(r["oa"])
    for r in metrics.read_csv("baselines_other.csv"):
        out.setdefault(r["model"], {})[r["dataset"]] = float(r["oa"])
    return out


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote docs/images/{name}")


# ------------------------------------------------------------------ 1. accuracy vs compute
def accuracy_vs_compute() -> None:
    eff = {r["model"]: r for r in metrics.read_csv("efficiency.csv")}
    base = baseline_oa()
    s, b = ours("small"), ours("base")
    mean_oa = {m: statistics.mean(base[m][d] for d in MEDMNIST9) for m in eff if m in base}
    mean_oa["HyMaC-Net-S"] = statistics.mean(s[d] for d in MEDMNIST9)
    mean_oa["HyMaC-Net-B"] = statistics.mean(b[d] for d in MEDMNIST9)
    family_color = {"CNN": NEUTRAL, "CNN-Transformer": THIRD, "SSM": BASE, "CNN-SSM (ours)": SMALL}
    family_label = {"CNN": "ResNet", "CNN-Transformer": "MedViT", "SSM": "MedMamba",
                    "CNN-SSM (ours)": "HyMaC-Net (ours)"}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.grid(True, axis="both")
    seen = set()
    for m, row in eff.items():
        x, y, p = float(row["gflops"]), mean_oa[m], float(row["params_m"])
        fam = row["family"]
        ax.scatter(x, y, s=18 + p * 9, color=family_color[fam], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=None if fam in seen else family_label[fam])
        seen.add(fam)
        dx, ha = (0.06, "left")
        if m in ("MedMamba-S",):
            dx, ha = (-0.06, "right")
        ax.annotate(f"{m}\n{p:g} M", (x, y), xytext=(x * (1 + dx), y), ha=ha, va="center",
                    fontsize=8, color=INK if "HyMaC" in m else INK2,
                    fontweight="bold" if "HyMaC" in m else "normal")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlim(0.9, 22)
    ax.set_xlabel("GFLOPs per 224×224 image (log scale)   ·   marker area ∝ parameters")
    ax.set_ylabel("Mean accuracy, 9 MedMNIST datasets (%)")
    ax.set_title("Accuracy against compute")
    ax.legend(loc="lower right", fontsize=8.5, markerscale=0.6)
    save(fig, "accuracy_vs_compute.png")


# ------------------------------------------------------------------ 2. per-dataset dot plot
def per_dataset() -> None:
    s, b = ours("small"), ours("base")
    base = baseline_oa()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.grid(True, axis="x")
    for i, d in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        compact = [(v[d], m) for m, v in base.items() if d in v and (m in COMPACT or d in ("CPN X-ray", "Kvasir"))]
        best = max(compact) if compact else None
        lo = min([s[d], b[d]] + ([best[0]] if best else []))
        hi = max([s[d], b[d]] + ([best[0]] if best else []))
        ax.plot([lo, hi], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        if best:
            ax.scatter(best[0], y, marker="|", s=170, color=INK2, linewidth=2, zorder=2)
            ax.annotate(best[1], (best[0], y), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7, color=MUTED)
        ax.scatter(b[d], y, s=60, color=BASE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(s[d], y, s=60, color=SMALL, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(list(reversed(ORDER)))
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(64, 100.5)
    ax.set_title("Accuracy on 12 medical imaging benchmarks")
    ax.scatter([], [], s=60, color=SMALL, label="HyMaC-Net-S (9.4 M)")
    ax.scatter([], [], s=60, color=BASE, label="HyMaC-Net-B (20.5 M)")
    ax.scatter([], [], marker="|", s=120, color=INK2, linewidth=2, label="best compact baseline as cited (≤ 26 M)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), fontsize=8.2, ncol=3)
    save(fig, "per_dataset_accuracy.png")


# ------------------------------------------------------------------ 3. seeds
def seeds() -> None:
    rows = metrics.read_csv("seeds_pathmnist.csv")
    stats = metrics.seed_summary()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, metric, label in ((axes[0], "oa", "Accuracy"), (axes[1], "auc", "AUC")):
        ax.grid(True, axis="y")
        h = [float(r[metric]) for r in rows if r["model"] == "HyMaC-Net-B"]
        r50 = [float(r[metric]) for r in rows if r["model"] == "ResNet-50"]
        for i, (a, c) in enumerate(zip(h, r50)):
            ax.plot([0, 1], [c, a], color=GRID, lw=1.6, zorder=1)
            ax.scatter(0, c, s=46, color=NEUTRAL, edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.scatter(1, a, s=46, color=BASE, edgecolor=SURFACE, linewidth=2, zorder=3)
        s = stats[metric]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ResNet-50", "HyMaC-Net-B"])
        ax.set_xlim(-0.35, 1.35)
        gain = f"+{100 * s['mean_difference']:.2f} pts" if metric == "oa" else f"+{s['mean_difference']:.3f}"
        ax.set_title(f"{label}: mean {gain}", fontsize=10)
        ax.set_ylabel(label)
    fig.suptitle("PathMNIST, 5 paired seeds: HyMaC-Net-B ahead on every seed (exact Wilcoxon one-sided p = 1/32)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "seed_robustness.png")


# ------------------------------------------------------------------ 4. ablation path
def ablation() -> None:
    rows = [r for r in metrics.read_csv("ablation_pathmnist.csv") if r["params_m"] and r["phase"] in "ABCD"]
    phase_color = {"A": NEUTRAL, "B": THIRD, "C": SMALL, "D": BASE}
    phase_name = {"A": "A · tokenisation", "B": "B · residual/SE scaffolding",
                  "C": "C · Mamba depth", "D": "D · regularisation & heads"}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.grid(True)
    for ph in "ABCD":
        pts = [r for r in rows if r["phase"] == ph]
        ax.scatter([float(r["params_m"]) for r in pts], [float(r["oa"]) for r in pts],
                   s=[30 + 22 * float(r["gmacs"]) for r in pts], color=phase_color[ph],
                   edgecolor=SURFACE, linewidth=1.5, label=phase_name[ph], zorder=3)
    for r in rows:
        if r["strategy"] in ("1", "12", "17", "19", "22"):
            note = {"1": "initial design", "12": "no RE blocks", "17": "3 Mamba blocks",
                    "19": "1 Mamba block", "22": "final HyMaC-Net-B"}[r["strategy"]]
            ax.annotate(f"S{r['strategy']}: {note}", (float(r["params_m"]), float(r["oa"])),
                        xytext=(6, 6), textcoords="offset points", fontsize=7.8,
                        color=INK if r["strategy"] == "22" else INK2,
                        fontweight="bold" if r["strategy"] == "22" else "normal")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 150])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "150"])
    ax.set_xlabel("Parameters (M, log scale)   ·   marker area ∝ GMACs")
    ax.set_ylabel("PathMNIST accuracy (%)")
    ax.set_title("Ablation path: 138 M → 20.5 M parameters, accuracy up")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "ablation_path.png")


# ------------------------------------------------------------------ 5. cross-dataset
def cross_dataset() -> None:
    rows = metrics.read_csv("cross_dataset.csv")
    cm = metrics.cross_dataset_matrix()
    m = metrics.confusion_metrics(cm["matrix"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    names = [r["model"] for r in rows]
    vals = [float(r["oa"]) for r in rows]
    colors = [BASE if "HyMaC" in n else NEUTRAL for n in names]
    a1.barh(range(len(names)), vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        a1.text(v + 1, i, f"{v:.2f} %", va="center", fontsize=8.5, color=INK)
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels(names)
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("Accuracy on unseen dataset (%)")
    a1.set_title("BRISC-25 → Brain Tumor MRI", fontsize=10)
    rec = [pc["recall"] * 100 for pc in m["per_class"]]
    a2.barh(range(len(rec)), rec, color=BASE, height=0.55)
    for i, (v, pc) in enumerate(zip(rec, m["per_class"])):
        a2.text(v + 1, i, f"{v:.0f} %  (n = {pc['support']})", va="center", fontsize=8.5, color=INK)
    a2.set_yticks(range(len(rec)))
    a2.set_yticklabels(cm["labels"])
    a2.invert_yaxis()
    a2.set_xlim(0, 125)
    a2.set_xticks([0, 25, 50, 75, 100])
    a2.set_xlabel("Recall per class, HyMaC-Net-B (%)")
    a2.set_title("Where it fails: glioma → no tumour", fontsize=10)
    fig.tight_layout()
    save(fig, "cross_dataset.png")


# ------------------------------------------------------------------ 6. efficiency
def efficiency() -> None:
    eff = metrics.read_csv("efficiency.csv")
    eff.sort(key=lambda r: float(r["gflops"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    for ax, key, label in ((axes[0], "params_m", "Parameters (M)"), (axes[1], "gflops", "GFLOPs")):
        vals = [float(r[key]) for r in eff]
        colors = [SMALL if r["model"] == "HyMaC-Net-S" else BASE if r["model"] == "HyMaC-Net-B" else NEUTRAL
                  for r in eff]
        ax.barh(range(len(eff)), vals, color=colors, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.015, i, f"{v:g}", va="center", fontsize=8, color=INK2)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(vals) * 1.15)
        ax.grid(True, axis="x")
    axes[0].set_yticks(range(len(eff)))
    axes[0].set_yticklabels([r["model"] for r in eff])
    fig.suptitle("Model size and compute", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "efficiency.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_vs_compute()
    per_dataset()
    seeds()
    ablation()
    cross_dataset()
    efficiency()


if __name__ == "__main__":
    main()
