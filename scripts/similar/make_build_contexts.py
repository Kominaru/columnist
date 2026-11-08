"""Pack the similar-charts index and the Lens code into small Docker build contexts for Portainer.

    python scripts/similar/make_build_contexts.py --data <folder with the exported data/similar> --out <empty folder>

Portainer's upload build takes at most ~10 MB per context, so the index (int8 embeddings, float16 weights, gzip) is cut into
chunks of <= 7.5 MB, one per build; the last build also carries the code, unpacks the chunks and tags the result
pattern-lens:latest. Build the contexts in order (ctx1.tar ... ctxN.tar), with the tag printed in ORDER.txt.
"""
import argparse
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNK = int(7.5 * 2 ** 20)


def pack_index(src: Path) -> bytes:
    emb = np.load(src / "embeddings.npy").astype(np.float32)
    scale = float(np.abs(emb).max()) / 127
    q = np.round(emb / scale).astype(np.int8)
    z = np.load(src / "model.npz")
    w = {k: (z[k].astype(np.float16) if z[k].dtype == np.float32 else z[k]) for k in z.files}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        def add(name, data):
            ti = tarfile.TarInfo(f"similar/{name}"); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
        b = io.BytesIO(); np.save(b, q); add("embeddings_q8.npy", b.getvalue())
        b = io.BytesIO(); np.save(b, np.float32(scale)); add("embeddings_q8_scale.npy", b.getvalue())
        b = io.BytesIO(); np.savez(b, **w); add("model.npz", b.getvalue())
        for f in ("mean.npy", "std.npy", "charts.csv", "info.json"):
            add(f, (src / f).read_bytes())
    return buf.getvalue()


def add_file(t: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = pack_index(Path(a.data))
    parts = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    order, n = [], len(parts)
    for k, part in enumerate(parts, 1):
        last = k == n
        base = "pattern-lens:latest" if k == 1 else f"lens-sim{k - 1}"
        tag = "pattern-lens:latest" if last else f"lens-sim{k}"
        lines = [f"FROM {base}", "USER root", f"COPY part{k} /app/parts/part{k}"]
        if last:
            lines += ["COPY osu_patterns /app/osu_patterns", "COPY service/viz /app/service/viz",
                      "RUN cat /app/parts/part* > /tmp/idx.tgz && rm -rf /app/parts && mkdir -p /app/models && "
                      "tar -xzf /tmp/idx.tgz -C /app/models && rm /tmp/idx.tgz && ls -l /app/models/similar",
                      "USER lens"]
        else:
            lines += ["USER lens"]
        with tarfile.open(out / f"ctx{k}.tar", "w") as t:
            add_file(t, "Dockerfile", ("\n".join(lines) + "\n").encode())
            add_file(t, f"part{k}", part)
            if last:
                for d in ("osu_patterns", "service/viz"):
                    t.add(ROOT / d, arcname=d, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
        order.append(f"ctx{k}.tar  ->  tag {tag}  ({(out / f'ctx{k}.tar').stat().st_size / 2**20:.1f} MB)")
    (out / "ORDER.txt").write_text("\n".join(order) + "\n")
    print("\n".join(order))


if __name__ == "__main__":
    main()
