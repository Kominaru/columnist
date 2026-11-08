"""Run the deep v2 variants one after another, then the comparison. For an unattended GPU machine.

    python scripts/run_v2_suite.py --device cuda
    python scripts/run_v2_suite.py --device cuda --only full ssl_stats
    python scripts/run_v2_suite.py --device cuda --only full3_s0 wide3_s0 full3_s1 wide3_s1

Each variant writes results/v2/deep_<name>_oof_f<k>.csv; a fold whose file exists is skipped, so
the suite can be restarted after an interruption. Logs go to logs/v2_<name>.txt.
"""
import argparse
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    "full": ["--ssl", "--stats", "--teacher"],
    "ssl_stats": ["--ssl", "--stats"],
    "scratch": ["--stage1-epochs", "0", "--stage2-epochs", "40", "--patience", "6"],
    "ssl": ["--ssl"],
}
# v3: the stronger teacher (boosted with transitions + relaxed labels), relaxed labels, mirror TTA;
# "wide" adds the boosted heads' inputs as a second path (train_deep_v2.py --wide)
V3 = ["--ssl", "--stats", "--teacher", "--teacher-file", "data/v3_teacher.csv.gz", "--relaxed", "--tta"]
# v4: teacher = mean of the v3 teacher and the MLP student (data/v4_teacher.csv.gz), longer stage 1
V4 = ["--ssl", "--stats", "--teacher", "--teacher-file", "data/v4_teacher.csv.gz", "--relaxed", "--tta",
      "--stage1-epochs", "16", "--patience", "6"]
# wide4r: v4 with the reading features and MinaCalc ratings in the wide path (data/extra_features.csv.gz)
# v5: teacher = boosted with reading + MinaCalc features, v3 teacher and the MLP student on the same inputs
V5 = [*V4[:4], "data/v5_teacher.csv.gz", *V4[5:]]
for s in range(3):
    VARIANTS[f"full3_s{s}"] = [*V3, "--seed", str(s)]
    VARIANTS[f"wide3_s{s}"] = [*V3, "--wide", "--seed", str(s)]
    VARIANTS[f"full4_s{s}"] = [*V4, "--seed", str(s)]
    VARIANTS[f"wide4_s{s}"] = [*V4, "--wide", "--wide-extra", "data/msd_features.csv.gz", "--seed", str(s)]
    VARIANTS[f"wide4r_s{s}"] = [*V4, "--wide", "--wide-extra", "data/extra_features.csv.gz", "--seed", str(s)]
# wide6: wide5 plus the visual event channels (train_deep_v2.py --visual): the encoder sees speed
# changes between notes, not only the speed at each note
for s in range(10):  # the last round: as many seeds as the GPU has time for
    VARIANTS[f"wide6_s{s}"] = [*V5, "--wide", "--wide-extra", "data/extra_features.csv.gz", "--visual", "--seed", str(s)]
    VARIANTS[f"wide5_s{s}"] = [*V5, "--wide", "--wide-extra", "data/extra_features.csv.gz", "--seed", str(s)]
    # wide5X: stage 1 still gains ~0.002 val mAP per epoch at epoch 14 of 16, so 30 stage-1 epochs
    # (patience stays 6: it applies to both stages)
    VARIANTS[f"wide5X_s{s}"] = [*V5[:7], "--stage1-epochs", "30", "--patience", "6", "--wide", "--wide-extra",
                                "data/extra_features.csv.gz", "--seed", str(s)]
    VARIANTS[f"full5X_s{s}"] = [*V5[:7], "--stage1-epochs", "30", "--patience", "6", "--seed", str(s)]
    VARIANTS[f"wide5XX_s{s}"] = [*V5[:7], "--stage1-epochs", "45", "--patience", "6", "--wide", "--wide-extra",
                                 "data/extra_features.csv.gz", "--seed", str(s)]
    # wide5L: a larger encoder (96 channels, 3 transformer layers), smaller batches to fit 8 GB
    VARIANTS[f"wide5L_s{s}"] = [*VARIANTS[f"wide5_s{s}"][:-2], "--channels", "96", "--n-tf", "3", "--max-tokens", "40000",
                                "--seed", str(s)]
    VARIANTS[f"full5_s{s}"] = [*V5, "--seed", str(s)]
    # full5U / wide5U: full5X / wide5X plus ~40k unranked (graveyard) 4K charts in every fold's stage 1
    # (scripts/list_unranked_mania.py, then prep_unranked.py; never labelled or tested)
    VARIANTS[f"full5U_s{s}"] = [*VARIANTS[f"full5X_s{s}"], "--extra", "data/unranked_4k_kept.csv"]
    VARIANTS[f"wide5U_s{s}"] = [*VARIANTS[f"wide5X_s{s}"], "--extra", "data/unranked_4k_kept.csv"]

# S: the 5X recipe retrained with the tournament-staff labels (relaxed files tagged v6, teacher v6)
VS = [*V5[:4], "data/v6_teacher.csv.gz", *V5[5:7], "--relaxed-tag", "v6", "--stage1-epochs", "30", "--patience", "6"]
for s in range(6):
    VARIANTS[f"fullS_s{s}"] = [*VS, "--seed", str(s)]
    VARIANTS[f"wideS_s{s}"] = [*VS, "--wide", "--wide-extra", "data/extra_features.csv.gz", "--seed", str(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=list(VARIANTS))
    ap.add_argument("--extra", default="", help="extra arguments for every train_deep_v2.py run")
    a = ap.parse_args()
    Path("logs").mkdir(exist_ok=True)
    done = []
    for name in a.only:
        todo = [k for k in range(5) if not Path(f"results/v2/deep_{name}_oof_f{k}.csv").exists()]
        if todo:
            cmd = [sys.executable, "scripts/train_deep_v2.py", "--name", name, "--device", a.device,
                   "--folds", ",".join(map(str, todo)), *VARIANTS[name], *a.extra.split()]
            print(" ".join(cmd), flush=True)
            with open(f"logs/v2_{name}.txt", "a") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"{name}: exit {r.returncode}", flush=True)
            if r.returncode:
                continue
        done.append(name)
    with open("logs/v2_compare.txt", "w") as log:
        subprocess.run([sys.executable, "scripts/compare_v2.py", "--deep", *done], stdout=log, stderr=subprocess.STDOUT)
    print(Path("logs/v2_compare.txt").read_text())


if __name__ == "__main__":
    main()
