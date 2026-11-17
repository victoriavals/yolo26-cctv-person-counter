"""Measure duplication in the raw CCTV-person export by pixel content, not by name.

Filename grouping alone is not trustworthy: Roboflow keeps the original stem, so
several genuinely different source images can share one stem (e.g. 44 files named
`youtube-0_jpg.rf.<hash>.jpg` that are not the same photograph). Conversely the
augmentation rotates frames, so an exact hash misses real duplicates.

Descriptor used here is rotation-tolerant: an HSV histogram over the non-fill
pixels (the rotation corner fill and letterbox bars are excluded), plus the
grayscale histogram. Rotating an image barely moves it; a different scene does.

Usage: python tools/audit_duplicates.py
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

RAW = "data/raw/cctv-person"
OUT = "reports/dup_audit.json"
SIM_THRESHOLD = 0.97  # calibrated below; printed distribution justifies it


def descriptor(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (128, 128))
        im = im.convert("RGB").resize((96, 96))
    except Exception:
        return None
    a = np.asarray(im, dtype=np.uint8)
    g = a.mean(axis=2)
    keep = g > 14  # drop rotation fill / letterbox
    if keep.sum() < 200:
        return None
    hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)[keep]
    h = np.histogramdd(
        hsv.astype(float), bins=(8, 4, 4),
        range=((0, 256), (0, 256), (0, 256)))[0].ravel()
    gh = np.histogram(g[keep], bins=32, range=(0, 256))[0].astype(float)
    v = np.concatenate([h / max(h.sum(), 1), gh / max(gh.sum(), 1)])
    n = np.linalg.norm(v)
    return v / n if n else None


def main():
    recs = []
    for split in ("train", "valid", "test"):
        d = json.load(open(f"{RAW}/{split}/_annotations.coco.json"))
        for i in d["images"]:
            recs.append((split, i["file_name"],
                         re.sub(r"\.rf\.[A-Za-z0-9]+\.jpg$", "", i["file_name"])))
    print(f"{len(recs)} files", file=sys.stderr)

    vecs, keep_recs = [], []
    for n, (split, fn, stem) in enumerate(recs):
        if n % 1000 == 0:
            print(f"  {n}/{len(recs)}", file=sys.stderr)
        v = descriptor(f"{RAW}/{split}/{fn}")
        if v is not None:
            vecs.append(v)
            keep_recs.append((split, fn, stem))
    V = np.stack(vecs)
    print(f"descriptors: {V.shape}", file=sys.stderr)

    by_stem = defaultdict(list)
    for idx, (split, fn, stem) in enumerate(keep_recs):
        by_stem[stem].append(idx)

    # --- calibration: similarity inside a stem group vs across random pairs
    rng = np.random.default_rng(0)
    within = []
    for idxs in by_stem.values():
        if len(idxs) < 2:
            continue
        for _ in range(min(6, len(idxs))):
            a, b = rng.choice(idxs, 2, replace=False)
            within.append(float(V[a] @ V[b]))
    ra = rng.integers(0, len(V), 4000)
    rb = rng.integers(0, len(V), 4000)
    across = (V[ra] * V[rb]).sum(axis=1)
    across = across[[keep_recs[i][2] != keep_recs[j][2] for i, j in zip(ra, rb)]]
    within = np.array(within)
    print("\nsimilarity within same filename stem: "
          f"p10={np.percentile(within,10):.3f} med={np.median(within):.3f}")
    print("similarity between different stems  : "
          f"med={np.median(across):.3f} p99={np.percentile(across,99):.3f} "
          f"p99.9={np.percentile(across,99.9):.3f}")
    print(f"threshold in use: {SIM_THRESHOLD}")
    print(f"  false-positive rate at threshold (unrelated pairs above it): "
          f"{(across > SIM_THRESHOLD).mean()*100:.2f}%")

    # --- cluster within each stem group; a cluster = one real photograph
    clusters = {}        # idx -> cluster id
    n_clusters = 0
    stems_multi_photo = 0
    for stem, idxs in by_stem.items():
        reps = []        # (cluster_id, representative idx)
        for i in idxs:
            hit = None
            for cid, r in reps:
                if float(V[i] @ V[r]) >= SIM_THRESHOLD:
                    hit = cid
                    break
            if hit is None:
                reps.append((n_clusters, i))
                hit = n_clusters
                n_clusters += 1
            clusters[i] = hit
        if len(reps) > 1:
            stems_multi_photo += 1

    cl_splits = defaultdict(set)
    cl_size = defaultdict(int)
    for i, cid in clusters.items():
        cl_splits[cid].add(keep_recs[i][0])
        cl_size[cid] += 1

    leaky = [c for c, s in cl_splits.items() if len(s) > 1]
    dup = [c for c, n in cl_size.items() if n > 1]
    leaked_files = sum(cl_size[c] for c in leaky)

    print("\n=== RESULT (by pixel content) ===")
    print(f"files                              : {len(keep_recs)}")
    print(f"filename stems                     : {len(by_stem)}")
    print(f"stems holding >1 distinct photo    : {stems_multi_photo}")
    print(f"DISTINCT PHOTOGRAPHS (clusters)    : {n_clusters}")
    print(f"photos with >1 augmented copy      : {len(dup)}")
    print(f"photos appearing in >1 split (LEAK): {len(leaky)}")
    print(f"files belonging to a leaking photo : {leaked_files} "
          f"({leaked_files/len(keep_recs)*100:.0f}% of the export)")
    for sp in ("train", "valid", "test"):
        n = sum(1 for i, cid in clusters.items()
                if keep_recs[i][0] == sp and cid in set(leaky))
        tot = sum(1 for r in keep_recs if r[0] == sp)
        print(f"  {sp:>5}: {n}/{tot} = {n/tot*100:.0f}% of its files share a photo with another split")

    json.dump({
        "files": len(keep_recs),
        "stems": len(by_stem),
        "stems_multi_photo": stems_multi_photo,
        "distinct_photos": n_clusters,
        "photos_with_copies": len(dup),
        "photos_leaking": len(leaky),
        "files_in_leaking_photos": leaked_files,
        "threshold": SIM_THRESHOLD,
        "assignments": {keep_recs[i][1]: int(c) for i, c in clusters.items()},
    }, open(OUT, "w"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()
