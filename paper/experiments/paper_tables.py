"""LaTeX tables from paper_out/*.json -> ../paper/tables/*.tex"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from paper_exp import FAMILIES  # noqa: E402

OUT = Path("paper_out")
T = Path("paper/tables")
T.mkdir(parents=True, exist_ok=True)
SL = ["RC", "LN", "HB", "SV"]


def j(name):
    return json.loads((OUT / name).read_text())


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&")


def tagname(t):
    return r"\utag{" + esc(t) + "}"


def data_table():
    s, t, c = j("slots.json"), j("tags.json"), j("categories.json")
    rows = [
        ("Tournament slots", "osu! wiki mappools", f"{s['tournaments']} tournaments",
         f"{s['labelled']:,}", f"{s['songs']:,} songs", r"4 classes (\RC, \LN, \HB, \SV)"),
        ("User tags", "ranked beatmapset pages", f"{t['sets']:,} sets",
         f"{t['diffs']:,}", f"{t['sets']:,} sets", "20 binary tags, masked"),
        ("Categories", "tags $\\cup$ pool sub-slots", "both of the above",
         f"{c['maps']:,}", f"{c['groups']:,} songs", "11 binary heads, masked"),
    ]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    keys = s["by_slot_keys"]  # {keys: {slot: n}}
    kline = "; ".join(f"{k}K: " + ", ".join(f"{sl} {keys[k].get(sl, 0)}" for sl in SL) for k in sorted(keys, key=int))
    (T / "data.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\caption{{Datasets. Charts are individual difficulties; CV groups are the units kept together in cross-validation.}}
\label{{tab:data}}
\begin{{tabular}}{{llllll}}
\toprule
Task & Source & Coverage & Charts & CV groups & Labels \\
\midrule
{body}
\bottomrule
\end{{tabular}}\\[2pt]
\parbox{{0.95\textwidth}}{{\footnotesize Slot labels by key count -- {kline}.}}
\end{{table*}}
""".replace(",", "{,}").replace("{{,}}", "{,}"))


def slots_main():
    s = j("slots.json")
    d = pd.read_csv(OUT / "slot_points.csv", index_col=0)
    lines = []
    for name, key in (("Rule baseline", "rule"), ("Gradient boosting", "model")):
        m = s[key]
        ci, fci = m["acc_ci"], m["mf1_ci"]
        perk = [np.mean(d.slot[d["keys"] == k] == d[("rule" if key == "rule" else "pred")][d["keys"] == k])
                for k in (4, 7)]
        lines.append(f"{name} & {pct(m['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}] & {f3(m['macro_f1'])} "
                     f"[{f3(fci[0])}, {f3(fci[1])}] & " + " & ".join(f3(m["f1"][c]) for c in SL)
                     + f" & {pct(perk[0])} & {pct(perk[1])}" + r" \\")
    (T / "slots_main.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\caption{{Slot classification on {s['labelled']:,} tournament charts (5-fold CV grouped by song). Brackets: 95\% cluster-bootstrap intervals over songs. The accuracy gain is {pct(s['model']['accuracy'] - s['rule']['accuracy'])} points [{pct(s['acc_diff_ci'][0])}, {pct(s['acc_diff_ci'][1])}].}}
\label{{tab:slots}}
\begin{{tabular}}{{lcccccccc}}
\toprule
& & & \multicolumn{{4}}{{c}}{{F1 per slot}} & \multicolumn{{2}}{{c}}{{accuracy by keys}} \\
\cmidrule(lr){{4-7}}\cmidrule(lr){{8-9}}
Method & Accuracy (\%) & Macro-F1 & \RC & \LN & \HB & \SV & 4K & 7K \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".replace(f"{s['labelled']:,}", f"{s['labelled']:,}".replace(",", "{,}")))


def slots_alt():
    s = j("slots.json")
    alt, abl = s["alternatives"], s["family_ablation"]
    rows = []
    order = [("HGB (ours)", "Gradient boosting, all 71 features"), ("Random forest", "Random forest, 500 trees"),
             ("Logistic regression", "Logistic regression (standardised)"),
             ("HGB, LN share + SV", "GB, LN share + 9 scroll-speed features"),
             ("HGB, LN share only", "GB, LN share only"), ("HGB, ungrouped CV", "GB, ungrouped (leaky) CV")]
    for k, lab in order:
        m = alt[k]
        rows.append(f"{lab} & {pct(m['accuracy'])} & {f3(m['macro_f1'])} & {f3(m['f1']['HB'])} & {f3(m['f1']['SV'])}" + r" \\")
    rows.append(r"\midrule")
    for fam in FAMILIES:
        m = abl[f"- {fam}"]
        rows.append(f"GB without {fam} ({len(FAMILIES[fam])}) & {pct(m['accuracy'])} & {f3(m['macro_f1'])} & "
                    f"{f3(m['f1']['HB'])} & {f3(m['f1']['SV'])}" + r" \\")
    rows.append(r"\midrule")
    for fam in FAMILIES:
        m = abl[f"only {fam}"]
        rows.append(f"GB with only {fam} & {pct(m['accuracy'])} & {f3(m['macro_f1'])} & "
                    f"{f3(m['f1']['HB'])} & {f3(m['f1']['SV'])}" + r" \\")
    (T / "slots_alt.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Slot model variants (grouped CV unless stated). Middle block: one feature family removed; bottom block: one family alone.}}
\label{{tab:slots-alt}}
\setlength\tabcolsep{{3pt}}
\begin{{tabular}}{{lcccc}}
\toprule
Variant & Acc.\ (\%) & Macro-F1 & \HB{{}} F1 & \SV{{}} F1 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def tags_table():
    t = j("tags.json")["tags"]
    rows = []
    for name, r in sorted(t.items(), key=lambda kv: -(kv[1]["ap"] - kv[1]["rule_ap"])):
        d = r["ap"] - r["rule_ap"]
        ci = r["diff_ci"]
        sig = r"$^\ast$" if ci[0] > 0 else (r"$^\dagger$" if ci[1] < 0 else "")
        best_m = r"\textbf{" + f3(r["ap"]) + "}" if r["ap"] >= r["rule_ap"] else f3(r["ap"])
        best_r = r"\textbf{" + f3(r["rule_ap"]) + "}" if r["rule_ap"] > r["ap"] else f3(r["rule_ap"])
        rows.append(f"{tagname(name)} & {r['n']:,} & {r['pos']} & {f3(r['prev'])} & {best_r} & {best_m} & "
                    f"{d:+.3f}{sig} & {f3(r['auc'])} & {f3(r['f1'])}".replace(",", "{,}") + r" \\")
    mean_m = np.mean([r["ap"] for r in t.values()])
    mean_r = np.mean([r["rule_ap"] for r in t.values()])
    rows.append(r"\midrule")
    rows.append(f"Mean over {len(t)} tags & & & & {f3(mean_r)} & {f3(mean_m)} & {mean_m - mean_r:+.3f} & "
                f"{f3(np.mean([r['auc'] for r in t.values()]))} & {f3(np.mean([r['f1'] for r in t.values()]))}" + r" \\")
    (T / "tags.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\caption{{User-tag heads on ranked charts (5-fold CV grouped by beatmapset). $n$: charts with a known label; AP: average precision; $\Delta$: head minus rule. $^\ast$/$^\dagger$: 95\% cluster-bootstrap interval of $\Delta$ above/below zero. AUC and F1 are for the learned head.}}
\label{{tab:tags}}
\begin{{tabular}}{{lrrrcccccc}}
\toprule
Tag & $n$ & pos. & prev. & Rule AP & Head AP & $\Delta$AP & AUC & F1 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def tags_ablation():
    t = j("tags.json")["tags"]
    full = np.array([r["ap"] for r in t.values()])
    variants = [
        ("Full head: $[\\mathbf{x},\\mathbf{h},\\mathbf{p}]$, masked labels", "ap"),
        ("Without slot probabilities $\\mathbf{p}$", "ap_no_slot"),
        ("Statistics $\\mathbf{x}$ only (no stacking)", "ap_stats_only"),
        ("Missing vote $=$ negative (train on all charts)", "ap_missing_as_neg"),
        ("Rule score $\\mathbf{h}_\\tau$ alone", "rule_ap"),
    ]
    rows = []
    for lab, k in variants:
        v = np.array([r[k] for r in t.values()])
        wins = int(np.sum(full > v + 1e-9)) if k != "ap" else None
        rows.append(f"{lab} & {f3(v.mean())} & {v.mean() - full.mean():+.3f} & "
                    f"{'--' if wins is None else f'{wins}/{len(t)}'}" + r" \\")
    (T / "tags_ablation.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Ablations of the tag heads, mean AP over the {len(t)} tags, always evaluated on the same masked labels. Last column: tags on which the full head is better.}}
\label{{tab:tags-abl}}
\setlength\tabcolsep{{3pt}}
\begin{{tabular}}{{lccc}}
\toprule
Variant & mean AP & vs.\ full & full better \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def categories_table():
    c = j("categories.json")["cats"]
    parent = {"Chordjack": "Rice", "Streams": "Rice", "Jumpstream/Handstream": "Rice", "Technical": "Rice",
              "Stamina": "Rice", "Release": "LN", "Coordination": "LN", "Density/Inverse": "LN",
              "Reading": "SV", "Memo": "SV", "TB": "--"}
    rows = []
    for name, r in c.items():
        d = r["ap"] - r["rule_ap"]
        ci = r["diff_ci"]
        sig = r"$^\ast$" if ci[0] > 0 else (r"$^\dagger$" if ci[1] < 0 else "")
        bm = r"\textbf{" + f3(r["ap"]) + "}" if r["ap"] >= r["rule_ap"] else f3(r["ap"])
        br = r"\textbf{" + f3(r["rule_ap"]) + "}" if r["rule_ap"] > r["ap"] else f3(r["rule_ap"])
        rows.append(f"{parent[name]} & {esc(name)} & {r['n']:,} & {r['pos']} ({r['pos_from_pools']}) & "
                    f"{f3(r['prev'])} & {br} & {bm} & {d:+.3f}{sig} & {f3(r['auc'])}".replace(",", "{,}") + r" \\")
    (T / "categories.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\caption{{Reviewer categories (5-fold CV grouped by song). pos.: positives, in brackets those from tournament sub-slots. Notation as in \cref{{tab:tags}}. Mixed Rice and Hybrid have no head (\cref{{sec:categories}}).}}
\label{{tab:cats}}
\begin{{tabular}}{{llrrcccccc}}
\toprule
Slot & Category & $n$ & pos.\ (pools) & prev. & Rule AP & Head AP & $\Delta$AP & AUC \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def features_table():
    rows = []
    for fam, cols in FAMILIES.items():
        rows.append(f"{fam.capitalize()} ({len(cols)}) & " + ", ".join(r"\texttt{" + esc(c) + "}" for c in cols) + r" \\")
    (T / "features.tex").write_text(r"""\begin{table*}[h]
\centering\footnotesize
\caption{The 71 chart statistics by family, as named in the released code.}
\label{tab:features}
\begin{tabular}{lp{0.78\textwidth}}
\toprule
Family & Features \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
