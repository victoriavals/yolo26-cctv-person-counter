"""Clean the CCTV-person Roboflow export.

Three things are wrong with the export (see README.md):
  1. 10062 files are only 4886 distinct photographs, inflated by baked-in
     augmentation (rotation, noise, exposure) applied before the export.
  2. Augmented copies of one photograph land in different splits: 75% of the
     test files share a photograph with train or valid.
  3. The `categories` table has 6 entries but every annotation uses id 2, whose
     name is a timestamp string.

This rebuilds it as: one copy per distinct photograph, single `person` class,
and a split grouped by source scene so no scene appears in two splits.

Which files are the same photograph is decided by pixel content, not by
filename -- 1507 filename stems hold more than one distinct photo, so grouping
by name would discard real images. Run tools/audit_duplicates.py first; this
script consumes its cluster assignments.

Usage: python tools/audit_duplicates.py && python tools/clean_cctv.py
"""
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import DROP_KINDS, kind_of  # noqa: E402

RAW = "data/raw/cctv-person"
OUT = "data/cctv-person"
AUDIT = "reports/dup_audit.json"
SPLIT_TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}

RF_SUFFIX = re.compile(r"\.rf\.[A-Za-z0-9]+\.jpg$")
FRAME_IDX = re.compile(r"^(.*?)[_-](\d{3,6})$")

# The export's 6 categories are a merge artifact. Three of them are people and
# collapse into `person`; two are shoes and must not become person boxes.
#   1 '0'                  84 boxes, h/w 2.65  -> person
#   2 'CCTV persons - ...' 37119 boxes, h/w 2.12 -> person
#   3 'Person'             57 boxes, h/w 3.54  -> person
#   4 'no_safetyshoes'     39 boxes, h/w 1.03, 0.32% of frame -> SHOES, drop
#   5 'safety_shoes'       21 boxes, h/w 1.39, 0.36% of frame -> SHOES, drop
# Category 0 ('person') carries no annotations at all.
# Only the valid split holds anything outside category 2, which is why a
# train-only check misses this.
PERSON_CATS = {0, 1, 2, 3}
DROP_CATS = {4, 5}


def original_name(file_name):
    """'Foo_00480_jpg.rf.abc123.jpg' -> 'Foo_00480_jpg'"""
    return RF_SUFFIX.sub("", file_name)


def scene_of(orig):
    """'Foo_Cut2_00480_jpg' -> 'Foo_Cut2'  (drops the frame index)"""
    stem = re.sub(r"_(png_)?jpg$", "", orig)
    m = FRAME_IDX.match(stem)
    return m.group(1) if m else stem


def fill_fraction(path):
    """Fraction of near-black pixels: rotation corner fill + letterbox bars.

    Used to pick the least-augmented copy of each original frame. Reads the
    JPEG at 1/8 scale via draft() so this stays fast over 10k files.
    """
    try:
        im = Image.open(path)
        im.draft("L", (80, 80))
        a = np.asarray(im.convert("L").resize((64, 64)), dtype=np.uint8)
    except Exception:
        return 1.0
    return float((a < 12).mean())


def load_raw():
    """-> {photo_cluster_id: [record, ...]} where a record is one augmented copy."""
    if not os.path.exists(AUDIT):
        sys.exit(f"missing {AUDIT} -- run tools/audit_duplicates.py first")
    cluster_of = json.load(open(AUDIT))["assignments"]
    load_raw.dropped_family = 0

    by_photo = defaultdict(list)
    for split in ("train", "valid", "test"):
        p = f"{RAW}/{split}/_annotations.coco.json"
        with open(p) as f:
            d = json.load(f)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for img in d["images"]:
            fn = img["file_name"]
            # youtube-* is webcam and selfie footage, not CCTV -- wrong angle,
            # distance and lens for an indoor-store camera, so it never enters
            # the dataset at all
            if kind_of(original_name(fn)) in DROP_KINDS:
                load_raw.dropped_family += 1
                continue
            # a file the audit could not describe (unreadable / all fill) keeps
            # its own bucket rather than being dropped
            key = cluster_of.get(fn, f"solo:{fn}")
            by_photo[key].append(
                {
                    "split_raw": split,
                    "file": f"{RAW}/{split}/{fn}",
                    "file_name": fn,
                    "stem": original_name(fn),
                    "w": img["width"],
                    "h": img["height"],
                    "boxes": [a["bbox"] for a in anns.get(img["id"], [])
                              if a["category_id"] in PERSON_CATS],
                    "dropped": sum(1 for a in anns.get(img["id"], [])
                                   if a["category_id"] in DROP_CATS),
                }
            )
    return by_photo


def pick_copies(by_photo):
    """One representative per distinct photograph: the copy with least fill area."""
    chosen = {}
    n = len(by_photo)
    for i, (key, copies) in enumerate(sorted(by_photo.items(), key=lambda kv: str(kv[0]))):
        if i % 500 == 0:
            print(f"  scoring {i}/{n}", file=sys.stderr)
        if len(copies) == 1:
            chosen[key] = copies[0]
            continue
        scored = [(fill_fraction(c["file"]), c["file_name"], c) for c in copies]
        scored.sort(key=lambda t: (round(t[0], 4), t[1]))
        chosen[key] = scored[0][2]
    return chosen


def density_bucket(counts):
    """Scene-level crowding, from the median boxes per photo in that scene."""
    m = float(np.median(counts))
    if m <= 2:
        return "d0-2"
    if m <= 5:
        return "d3-5"
    if m <= 10:
        return "d6-10"
    return "d10+"


def assign_splits(scene_sizes, scene_stratum):
    """Greedy allocation run independently inside each stratum.

    A plain global greedy keeps the 80/10/10 ratio but lets domain and density
    drift: the previous split landed 18% indoor in test against 36% overall,
    and a test median of 5 people per frame against train's 2. Evaluating an
    indoor-store model on that is measuring the wrong thing.

    Stratifying on (domain kind x density bucket) holds both proportions
    steady across splits. Scenes still move as whole units, so the no-leakage
    guarantee is unchanged.
    """
    by_stratum = defaultdict(list)
    for scene, size in scene_sizes.items():
        by_stratum[scene_stratum[scene]].append((scene, size))

    out = {}
    have = Counter()
    # Targets accumulate per stratum but the deficit is tracked globally, so a
    # stratum that overshoots one split is compensated by the next. Purely
    # per-stratum quotas balance each stratum but let the global ratio drift
    # (measured: 77/12/11 instead of 80/10/10 across 14 strata).
    target = {s: 0.0 for s in SPLIT_TARGET}
    for stratum in sorted(by_stratum, key=str):
        scenes = sorted(by_stratum[stratum], key=lambda kv: (-kv[1], kv[0]))
        for s, f in SPLIT_TARGET.items():
            target[s] += f * sum(sz for _, sz in scenes)
        for scene, size in scenes:
            s = max(SPLIT_TARGET, key=lambda k: (target[k] - have[k]) / max(target[k], 1))
            out[scene] = s
            have[s] += size
    return out, dict(have)


# Known limitation, kept deliberately: the `semi` family (entrances, 3.0% of the
# data, 142 photos) lands entirely in train. Forcing it into val and test by
# reserving each stratum's smallest scenes was tried and made things much worse
# -- test swung to 60.9% outdoor and 1.3% unknown. Balanced indoor share and an
# exact 80/10/10 matter more than covering a 3% family, so the deficit chase
# stays as-is and entrances simply are not measured on the held-out splits.


def main():
    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/annotations", exist_ok=True)

    by_photo = load_raw()
    n_files = sum(len(v) for v in by_photo.values())
    print(f"raw: {n_files} files ({load_raw.dropped_family} dropped as non-CCTV) "
          f"-> {len(by_photo)} distinct photographs")

    chosen = pick_copies(by_photo)

    scene_of_orig = {k: scene_of(rec["stem"]) for k, rec in chosen.items()}
    scene_sizes = Counter(scene_of_orig.values())

    # stratum per scene: domain kind (majority of its photos) x density bucket.
    # kind is read off the original stem, never the scene name -- scene_of()
    # strips trailing frame indices, which breaks patterns like ^image[_-]?\d.
    scene_boxes, scene_kinds = defaultdict(list), defaultdict(Counter)
    for k, rec in chosen.items():
        sc = scene_of_orig[k]
        scene_boxes[sc].append(len(rec["boxes"]))
        scene_kinds[sc][kind_of(rec["stem"])] += 1
    scene_stratum = {
        sc: (scene_kinds[sc].most_common(1)[0][0], density_bucket(scene_boxes[sc]))
        for sc in scene_sizes
    }

    split_of_scene, have = assign_splits(scene_sizes, scene_stratum)
    print(f"scenes: {len(scene_sizes)} in {len(set(scene_stratum.values()))} strata  "
          f"split sizes: {have}")

    coco = {
        s: {
            "info": {"description": "CCTV-person, deduplicated and re-split by scene"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        for s in SPLIT_TARGET
    }
    img_id = defaultdict(int)
    ann_id = defaultdict(int)
    dropped = 0
    manifest = []
    # Windows is case-insensitive: 4 originals differ from another only by case
    # (e.g. 1_PNG_jpg vs 1_png_jpg) and would silently overwrite each other.
    taken = set()

    for key in sorted(chosen, key=str):
        rec = chosen[key]
        scene = scene_of_orig[key]
        split = split_of_scene[scene]
        stem = rec["stem"]
        dst_name = stem + ".jpg"
        n = 2
        while dst_name.lower() in taken:
            dst_name = f"{stem}__{n}.jpg"
            n += 1
        taken.add(dst_name.lower())
        shutil.copyfile(rec["file"], f"{OUT}/images/{dst_name}")

        img_id[split] += 1
        iid = img_id[split]
        coco[split]["images"].append(
            {"id": iid, "file_name": dst_name, "width": rec["w"], "height": rec["h"]}
        )
        for x, y, w, h in rec["boxes"]:
            if w <= 1 or h <= 1:          # 2 degenerate boxes exist in the export
                dropped += 1
                continue
            ann_id[split] += 1
            coco[split]["annotations"].append(
                {
                    "id": ann_id[split],
                    "image_id": iid,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        manifest.append(
            {
                "file": dst_name,
                "scene": scene,
                "split": split,
                # carried here so evaluate.py can break metrics down by domain
                # without re-deriving the family patterns
                "kind": scene_stratum[scene][0],
                "density_bucket": scene_stratum[scene][1],
                "boxes": len(rec["boxes"]),
                "copies_in_export": len(by_photo[key]),
                "split_in_export": rec["split_raw"],
            }
        )

    for s, d in coco.items():
        with open(f"{OUT}/annotations/{s}.json", "w") as f:
            json.dump(d, f)
        print(f"{s:>6}: {len(d['images'])} images, {len(d['annotations'])} boxes")
    shoes = sum(rec["dropped"] for rec in chosen.values())
    print(f"dropped {dropped} degenerate boxes, {shoes} shoe boxes (kelas 4/5)")

    with open(f"{OUT}/manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0]))
        wr.writeheader()
        wr.writerows(manifest)

    write_yolo(coco)


def write_yolo(coco):
    """Ultralytics-style mirror: images/<split> symlink-free, labels/<split>/*.txt."""
    for s, d in coco.items():
        os.makedirs(f"{OUT}/yolo/labels/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/yolo/images/{s}", exist_ok=True)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a["bbox"])
        for img in d["images"]:
            stem = os.path.splitext(img["file_name"])[0]
            W, H = img["width"], img["height"]
            lines = []
            for x, y, w, h in anns.get(img["id"], []):
                lines.append(
                    f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                )
            with open(f"{OUT}/yolo/labels/{s}/{stem}.txt", "w") as f:
                f.write("\n".join(lines))
            shutil.copyfile(
                f"{OUT}/images/{img['file_name']}", f"{OUT}/yolo/images/{s}/{img['file_name']}"
            )
    with open(f"{OUT}/yolo/data.yaml", "w") as f:
        root = os.path.abspath(f"{OUT}/yolo").replace("\\", "/")
        f.write("# CCTV-person, sudah dibersihkan: dedup per foto, split per-scene, satu kelas.\n"
                "# Ganti `path` kalau folder ini dipindah.\n"
                f"path: {root}\n"
                "train: images/train\nval: images/val\ntest: images/test\n\n"
                "nc: 1\nnames: [person]\n")
    print("yolo mirror written")


if __name__ == "__main__":
    main()
