"""Convert joint_plain.final.pt of the release package to joint.npz + joint.json for torch-free serving.
usage: python scripts/export_joint_npz.py joint_plain.final.pt config.json calibration_joint_plain.json out_dir"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ck_path, cfg_path, cal_path, out = sys.argv[1:5]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
cfg = json.load(open(cfg_path)); cal = json.load(open(cal_path))
arrs = {f"enc.{k}": v.float().numpy() for k, v in ck["model"].items() if not k.startswith(("tok_head", "gap_head", "proj", "rank_head", "order_head", "stat_head", "rate_head"))}
arrs.update({f"head.{k}": v.float().numpy() for k, v in ck["head"].items()})
arrs["layers"], arrs["d"] = np.array(ck["layers"]), np.array(ck["d"])
arrs["heads"] = np.array(len(ck["model"]["slopes"]))
out = Path(out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "joint.npz", **arrs)
(out / "joint.json").write_text(json.dumps({"tasks": ck["tasks"], "thresholds": {k: v["threshold"] for k, v in cfg["thresholds"].items()}, "len_feat_stats": ck["len_feat_stats"],
                                           "topk": ck["topk"], "calibration": {"intercept": cal["linear"]["intercept"], "slope": cal["linear"]["slope"], "isotonic": cal["isotonic"]}}))
print("wrote", out)
