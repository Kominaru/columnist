"""Tables, per-task figure data and number macros for the categories / user-tags part of the paper.

    python paper/experiments/v2_tables.py [--boot 500]

Run from the repository root after the v2 runs (results/v2, results/v5_x). Reads the out-of-fold
predictions of the final configuration and writes
  paper/tables/v2_data.tex         corpus and label counts
  paper/tables/v2_categories.tex   per-task AP: rule, baseline, the four components, the ensemble
  paper/tables/v2_tags.tex
  paper/tables/v2_components.tex   each component alone and the ensemble without it
  paper/fig/tikz/pertask.tex       per-task AP figure (pgfplots)
  paper/numbers.tex                \\newcommand macros used in the text (with --boot: intervals)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from osu_patterns import categories  # noqa: E402

spec = importlib.util.spec_from_file_location("ens", ROOT / "scripts/ensemble_v2.py")
ens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ens)

PAPER = ROOT / "paper"
BOOSTED = ["gbm", "gbm+win", "gbm+win+pat", "gbm+win+relax", "gbm+win+pat+relax"]
SV_SPEC = "oof:gbm+win+pat+x+relax"


def runs(prefix, res):
    """Completed deep runs (all five folds) whose name starts with prefix_s."""
    out = []
    for s in range(20):
        if len(list(res.glob(f"deep_{prefix}_s{s}_oof_f*.csv"))) == 5:
            out.append(f"{prefix}_s{s}")
    return out


def ap_of(Y, P, cols):
    r = {}
    for t in cols:
        m = Y[t].notna() & P[t].notna()
        r[t] = aps(Y.loc[m, t].astype(int), P.loc[m, t])
    return pd.Series(r)


def tex_name(t):
    t = t[4:] if t.startswith("tag:") else t
    return t.replace("&", r"\&").replace("_", r"\_")


def per_task_table(Y, cols, preds, label, caption, rel):
    names = list(preds)
    A = pd.DataFrame({n: ap_of(Y, preds[n], cols) for n in names})
    lines = [r"\begin{table*}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lrr" + "c" * len(names) + "}", r"\toprule",
             r"& & & & & \multicolumn{4}{c}{components} & \\ \cmidrule(lr){6-9}",
             r"Task & pos. & known & " + " & ".join(rel[n] for n in names) + r" \\", r"\midrule"]
    for t in cols:
        y = Y[t].dropna()
        best = A.loc[t, names[1:]].max()
        cells = []
        for n in names:
            v = A.loc[t, n]
            s = f"{v:.3f}"
            cells.append(rf"\textbf{{{s}}}" if n != names[0] and abs(v - best) < 5e-4 else s)
        sv = r"$^\dagger$" if t in ens.SV_TASKS else ""
        name = rf"\utag{{{tex_name(t)}}}" if t.startswith("tag:") else tex_name(t)
        lines.append(rf"{name}{sv} & {int(y.sum())} & {len(y)} & " + " & ".join(cells) + r" \\")
    mean = A.mean()
    best = mean[names[1:]].max()
    lines += [r"\midrule", r"Mean & & & " + " & ".join(
        (rf"\textbf{{{mean[n]:.3f}}}" if n != names[0] and abs(mean[n] - best) < 5e-4 else f"{mean[n]:.3f}") for n in names)
        + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n", A


def boot(Y, E, base, grp, cols, n, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(grp)
    gi = {g: np.where(grp == g)[0] for g in ug}
    yv = {t: Y[t].to_numpy() for t in cols}
    ev = {t: E[t].to_numpy() for t in cols}
    bv = {t: base[t].to_numpy() for t in cols}

    def diff(idx):
        out = []
        for t in cols:
            y, e, b = yv[t][idx], ev[t][idx], bv[t][idx]
            m = ~np.isnan(y) & ~np.isnan(e) & ~np.isnan(b)
            if y[m].sum() == 0:
                continue
            out.append(aps(y[m], e[m]) - aps(y[m], b[m]))
        return float(np.mean(out))

    full = diff(np.arange(len(Y)))
    bs = np.array([diff(np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])) for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return full, lo, hi, float(np.mean(bs >= 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--plain", default="full5X")
    ap.add_argument("--wide", default="wide5X")
    a = ap.parse_args()
    d, res = ROOT / "data", ROOT / "results/v2"
    Y = pd.read_csv(d / "v2_labels.csv.gz", index_col=0)
    slot = Y.pop("slot")
    folds = pd.read_csv(d / "v2_folds.csv", index_col=0).reindex(Y.index)
    oof = pd.concat([pd.read_csv(p, index_col=0, header=[0, 1]) for p in (res / "oof.csv", ROOT / "results/v5_x/oof.csv")], axis=1)
    oof = oof.loc[:, ~oof.columns.duplicated()].reindex(Y.index)
    plain, wide = runs(a.plain, res), runs(a.wide, res)
    print("plain runs:", plain, "\nwide runs:", wide, flush=True)
    load = lambda s: ens.load(s, Y, res, oof, BOOSTED)  # noqa: E731
    comp = {
        "boosted": load(f"boosted|{SV_SPEC}"),
        "mlp": load("mlp:v5"),
        "plain": load("deep:" + "+".join(plain)),
        "wide": load("deep:" + "+".join(wide)),
    }
    E = sum(comp.values()) / len(comp)
    rule, base = oof["rule"][Y.columns], oof["gbm+win"][Y.columns]
    cats = [c for c in Y.columns if c in categories.NAMES]
    tags = [c for c in Y.columns if c.startswith("tag:")]
    rel = {"rule": "rule", "base": "gbm+win", "boosted": "boosted", "mlp": "MLP", "plain": "deep",
           "wide": "wide\\,\\&\\,deep", "ens": "ensemble"}
    preds = {"rule": rule, "base": base, **comp, "ens": E}
    note = (r"Average precision on the strict out-of-fold labels (song-grouped 5-fold CV over all charts). "
            r"pos./known: positive and labelled charts. gbm+win is the reference boosted head; the four components are averaged "
            r"into the ensemble. Bold: best learned model. $^\dagger$\,\SV{} heads, for which the boosted component uses the reading "
            r"and MinaCalc features.")
    t_cat, A_cat = per_task_table(Y, cats, preds, "tab:v2-cats", "Reviewer categories. " + note, rel)
    t_tag, A_tag = per_task_table(Y, tags, preds, "tab:v2-tags", "User tags. " + note, rel)
    (PAPER / "tables/v2_categories.tex").write_text(t_cat)
    (PAPER / "tables/v2_tags.tex").write_text(t_tag)

    # components alone and left out
    def mean_ap(P, cols):
        return float(ap_of(Y, P, cols).mean())
    rows = [("gbm+win (reference)", base)]
    rows += [(f"{rel[k]} alone", v) for k, v in comp.items()]
    rows += [("ensemble of the four", E)]
    for k in comp:
        rest = [v for kk, v in comp.items() if kk != k]
        rows.append((f"\\quad without {rel[k]}", sum(rest) / 3))
    nosv = dict(comp, boosted=load("boosted"))
    rows.append((r"\quad \SV{} heads: boosted without reading", sum(nosv.values()) / 4))
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Mean AP of each component alone and of the ensemble without it. The four components are "
             r"averaged with equal weights; nothing is fitted on the evaluation folds.}",
             r"\label{tab:v2-components}", r"\begin{tabular}{lcc}", r"\toprule", r"Model & Categories & Tags \\", r"\midrule"]
    comp_means = {}
    for name, P in rows:
        mc, mt = mean_ap(P, cats), mean_ap(P, tags)
        comp_means[name] = (mc, mt)
        lines.append(f"{name} & {mc:.3f} & {mt:.3f} \\\\")
        if name.startswith("ensemble") or name.startswith("gbm"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "tables/v2_components.tex").write_text("\n".join(lines) + "\n")

    # corpus and labels
    pools = set(pd.read_csv(d / "mappools.csv")["beatmap_id"].dropna().astype(int))
    ranked = set(pd.read_csv(d / "all_ranked_ids.csv")["beatmap_id"].astype(int))
    ids = set(Y.index)
    n_pool, n_ranked = len(ids & pools), len(ids & ranked - pools)
    n_other = len(ids) - n_pool - n_ranked
    lab_c, lab_t = Y[cats].notna().any(axis=1), Y[tags].notna().any(axis=1)
    Yr = pd.read_csv(d / "v2_labels_relaxed.csv.gz", index_col=0).reindex(Y.index)
    relax_add = int((Yr[Y.columns].notna() & Y.notna().eq(False)).sum().sum())
    relax_pos = int(((Yr[Y.columns] == 1) & Y.isna()).sum().sum())
    data_tex = rf"""\begin{{table}}[t]
\centering\small
\caption{{The chart corpus of the category and tag experiments. Every chart is assigned to one of five folds by its song; a model for fold $k$ never sees a chart of a fold-$k$ song.}}
\label{{tab:v2-data}}
\begin{{tabular}}{{lr}}
\toprule
Charts in the corpus & \num{{{len(Y)}}} \\
\quad in a tournament pool & \num{{{n_pool}}} \\
\quad ranked, not pooled & \num{{{n_ranked}}} \\
Songs (fold groups) & \num{{{folds['group'].nunique()}}} \\
\midrule
Charts with $\ge1$ strict category label & \num{{{int(lab_c.sum())}}} \\
Charts with $\ge1$ strict tag label & \num{{{int(lab_t.sum())}}} \\
Charts with no strict label & \num{{{int((~lab_c & ~lab_t).sum())}}} \\
Relaxed (chart, task) labels added & \num{{{relax_add}}} \\
\quad of which positive & \num{{{relax_pos}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (PAPER / "tables/v2_data.tex").write_text(data_tex)

    # per-task figure (pgfplots): rule, reference and ensemble AP per task
    order = cats + tags
    ys, ticks, lines = {}, [], []
    y = 0
    for t in reversed(order):
        ys[t] = y
        y += 1 if not (t == tags[0]) else 1.8  # gap between the two benchmarks
    for t in order:
        A = A_cat if t in cats else A_tag
        ticks.append((ys[t], tex_name(t)))
        lines.append((ys[t], A.loc[t, "rule"], A.loc[t, "base"], A.loc[t, "ens"]))
    def coords(i):
        return " ".join(f"({v[i]:.4f},{v[0]:.1f})" for v in lines)
    seg = "\n".join(f"\\draw[model!45, line width=1.2pt] (axis cs:{b:.4f},{yy:.1f}) -- (axis cs:{e:.4f},{yy:.1f});"
                     for yy, r, b, e in lines)
    fig = (
        "\\begin{tikzpicture}\n\\begin{axis}[scale only axis, width=0.6\\columnwidth, height=9.6cm, xmin=0, xmax=1, ymin=-0.8, "
        f"ymax={max(ys.values()) + 0.8:.1f},\n"
        "  ytick={" + ",".join(f"{yy:.1f}" for yy, _ in ticks) + "},\n"
        "  yticklabels={" + ",".join("{" + n + "}" for _, n in ticks) + "},\n"
        "  yticklabel style={font=\\scriptsize}, xticklabel style={font=\\scriptsize}, xlabel={average precision}, "
        "xlabel style={font=\\scriptsize}, xmajorgrids, grid style={faint}, axis line style={muted!60}, tick style={muted!60},\n"
        "  legend style={font=\\scriptsize, at={(0.02,0.985)}, anchor=north west, draw=none, fill=white, fill opacity=0.85, text opacity=1}, "
        "legend cell align=left]\n"
        + seg + "\n"
        "\\addplot[only marks, mark=o, mark size=1.5pt, muted] coordinates {" + coords(1) + "};\n\\addlegendentry{rule}\n"
        "\\addplot[only marks, mark=*, mark size=1.6pt, rule] coordinates {" + coords(2) + "};\n\\addlegendentry{gbm+win}\n"
        "\\addplot[only marks, mark=*, mark size=1.8pt, model] coordinates {" + coords(3) + "};\n\\addlegendentry{ensemble}\n"
        f"\\draw[muted!50, densely dashed] (axis cs:0,{ys[tags[0]] + 0.9:.1f}) -- (axis cs:1,{ys[tags[0]] + 0.9:.1f});\n"
        f"\\node[font=\\scriptsize\\bfseries, anchor=east, text=muted] at (axis cs:0.99,{ys[cats[-1]] - 0.05:.1f}) {{categories}};\n"
        f"\\node[font=\\scriptsize\\bfseries, anchor=east, text=muted] at (axis cs:0.99,{ys[tags[-1]] - 0.05:.1f}) {{user tags}};\n"
        "\\end{axis}\n\\end{tikzpicture}\n")
    (PAPER / "fig/tikz/pertask.tex").write_text(fig)

    # macros
    mac = {}
    mc, mt = A_cat.mean(), A_tag.mean()
    for k, key in (("rule", "Rule"), ("base", "Base"), ("boosted", "Boost"), ("mlp", "Mlp"), ("plain", "Deep"),
                   ("wide", "Wide"), ("ens", "Ens")):
        mac[f"cat{key}"] = f"{mc[k]:.3f}"
        mac[f"tag{key}"] = f"{mt[k]:.3f}"
    mac["catGain"], mac["tagGain"] = f"{mc['ens'] - mc['base']:+.3f}", f"{mt['ens'] - mt['base']:+.3f}"
    mac["catBeatsBase"] = str(int((A_cat["ens"] > A_cat["base"]).sum()))
    mac["tagBeatsBase"] = str(int((A_tag["ens"] > A_tag["base"]).sum()))
    mac["nPlain"], mac["nWide"] = str(len(plain)), str(len(wide))
    mac["nCharts"] = f"{len(Y):,}".replace(",", "{,}")
    mac["nSongs"] = f"{folds['group'].nunique():,}".replace(",", "{,}")
    mac["nUnlab"] = f"{int((~lab_c & ~lab_t).sum()):,}".replace(",", "{,}")
    nosv_c, nosv_t = comp_means[r"\quad \SV{} heads: boosted without reading"]
    mac["catNoSv"], mac["tagNoSv"] = f"{nosv_c:.3f}", f"{nosv_t:.3f}"
    for t, key in (("Memo", "Memo"), ("Reading", "Reading")):
        mac[f"ap{key}Base"] = f"{A_cat.loc[t, 'base']:.2f}"
        mac[f"ap{key}Ens"] = f"{A_cat.loc[t, 'ens']:.2f}"
        P = sum(nosv.values()) / 4
        mac[f"ap{key}NoSv"] = f"{ap_of(Y, P, [t]).iloc[0]:.2f}"
        mac[f"ap{key}BoostNoSv"] = f"{ap_of(Y, load('boosted'), [t]).iloc[0]:.2f}"
        mac[f"ap{key}BoostSv"] = f"{ap_of(Y, comp['boosted'], [t]).iloc[0]:.2f}"
    if a.boot:
        grp = folds["group"].fillna(pd.Series(Y.index.astype(str), index=Y.index)).to_numpy()
        for name, cols in (("cat", cats), ("tag", tags)):
            full, lo, hi, share = boot(Y, E, base, grp, cols, a.boot)
            print(name, full, lo, hi, share, flush=True)
            mac[f"{name}Lo"], mac[f"{name}Hi"] = f"{lo:+.3f}", f"{hi:+.3f}"
            mac[f"{name}Share"] = f"{share:.2f}"
        mac["nBoot"] = str(a.boot)
    old = {}
    p = PAPER / "numbers.tex"
    if p.exists() and not a.boot:  # keep earlier bootstrap macros
        for line in p.read_text().splitlines():
            if line.startswith(r"\newcommand"):
                k = line.split("{")[1].lstrip("\\").rstrip("}")
                old[k] = line
    out = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    out += [v for k, v in old.items() if k not in mac]
    p.write_text("% generated by paper/experiments/v2_tables.py\n" + "\n".join(out) + "\n")
    print(open(p).read())
    print(pd.DataFrame(comp_means, index=["cat", "tag"]).T.round(4).to_string())


if __name__ == "__main__":
    main()
