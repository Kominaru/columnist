"""Figures for the paper from paper_out/*.json and *.csv. Writes PDFs to paper_out/fig/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

OUT = Path("paper_out")
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)
MODEL, RULE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.35  # one column, inches
SLOTS = ["RC", "LN", "HB", "SV"]
NAMES = {"RC": "Rice", "LN": "Long note", "HB": "Hybrid", "SV": "Scroll speed"}


def ln_share():
    d = pd.read_csv(OUT / "slot_points.csv", index_col=0)
    fig, axes = plt.subplots(4, 1, figsize=(COL, 2.6), sharex=True)
    bins = np.linspace(0, 1, 51)
    for ax, s in zip(axes, SLOTS):
        x = d.ln_ratio[d.slot == s]
        ax.hist(x, bins=bins, color=MODEL, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        xt, ha = (0.13, "left") if s == "LN" else (0.99, "right")
        ax.text(xt, 0.82, f"{NAMES[s]} (n={len(x)}, median {x.median():.2f})", transform=ax.transAxes,
                ha=ha, va="top", color=INK, fontsize=7)
        for v in (0.111, 0.367):
            ax.axvline(v, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    axes[-1].set_xlabel("LN share (fraction of notes that are long notes)")
    axes[0].text(0.055, 1.02, "rule RC", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=6, color=MUTED)
    axes[0].text(0.24, 1.02, "rule HB", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=6, color=MUTED)
    axes[0].text(0.70, 1.02, "rule LN", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=6, color=MUTED)
    fig.savefig(FIG / "ln_share.pdf")


def confusion():
    r = json.loads((OUT / "slots.json").read_text())
    from sklearn.metrics import confusion_matrix
    d = pd.read_csv(OUT / "slot_points.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75))
    for ax, col, title in ((axes[0], "rule", "Rule baseline"), (axes[1], "pred", "Gradient boosting")):
        cm = confusion_matrix(d.slot, d[col], labels=SLOTS)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), SLOTS)
        ax.set_yticks(range(4), SLOTS)
        ax.set_xlabel("predicted")
        acc = r["rule" if col == "rule" else "model"]["accuracy"]
        ax.set_title(f"{title} (acc. {acc:.3f})")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("tournament slot")
    fig.savefig(FIG / "confusion.pdf")


def dumbbell(items, fname, height):
    """items: list of (label, rule_ap, model_ap, prevalence)."""
    items = sorted(items, key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(COL, height))
    y = np.arange(len(items))
    for i, (lab, h, m, prev) in enumerate(items):
        ax.plot([h, m], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
        ax.plot([prev], [i], marker="|", color=MUTED, ms=6, mew=0.8, zorder=2)
    ax.scatter([t[1] for t in items], y, s=16, color=RULE, zorder=3, label="Rule", edgecolor="white", lw=0.5)
    ax.scatter([t[2] for t in items], y, s=16, color=MODEL, zorder=4, label="Learned head", edgecolor="white", lw=0.5)
    ax.plot([], [], marker="|", ls="none", color=MUTED, label="Chance (prevalence)")
    ax.set_yticks(y, [t[0] for t in items])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average precision (grouped 5-fold CV)")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.1)
    fig.savefig(FIG / fname)


def tags_fig():
    r = json.loads((OUT / "tags.json").read_text())["tags"]
    dumbbell([(t, v["rule_ap"], v["ap"], v["prev"]) for t, v in r.items()], "tags_ap.pdf", 3.3)


def cats_fig():
    r = json.loads((OUT / "categories.json").read_text())["cats"]
    dumbbell([(t, v["rule_ap"], v["ap"], v["prev"]) for t, v in r.items()], "categories_ap.pdf", 2.0)


def pr_fig():
    c = json.loads((OUT / "category_curves.json").read_text())
    show = ["Jumpstream/Handstream", "Coordination", "Reading", "Memo"]
    fig, axes = plt.subplots(1, 4, figsize=(2 * COL + 0.3, 1.55), sharey=True)
    for ax, n in zip(axes, show):
        y = np.array(c[n]["y"])
        for key, color, lab in (("h", RULE, "Rule"), ("p", MODEL, "Learned head")):
            p, rr, _ = precision_recall_curve(y, np.array(c[n][key]))
            ax.plot(rr, p, color=color, lw=1.4, label=lab)
        ax.axhline(y.mean(), color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(f"{n} ({y.sum()} pos.)")
        ax.set_xlabel("recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID, lw=0.4)
    axes[0].set_ylabel("precision")
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig(FIG / "pr_curves.pdf")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        globals()[f]()
