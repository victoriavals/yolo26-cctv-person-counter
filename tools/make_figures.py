"""Render the figures and chart data for the CCTV-person audit report.

Outputs PNG panels to reports/figures/ and chart data to reports/stats.json.

Usage: python tools/make_figures.py
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import family_of  # noqa: E402

CCTV = "data/cctv-person"
RAW_CCTV = "data/raw/cctv-person"
FIG = "reports/figures"

INK = (250, 250, 248)
PANEL_BG = (20, 20, 19)
BAD = (227, 73, 72)
GOOD = (27, 175, 122)
NEUTRAL = (150, 150, 145)
BOXBOX = (232, 232, 228)

def font(size, bold=False):
    for name in (("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_boxes(im, boxes, width=2, scale=1.0):
    d = ImageDraw.Draw(im)
    for x, y, w, h in boxes:
        d.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale],
                    outline=BOXBOX, width=width)
    return im


def caption(im, text, sub=None, color=INK):
    hpad = 40 if sub else 26
    out = Image.new("RGB", (im.width, im.height + hpad), PANEL_BG)
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((6, im.height + 4), text, fill=color, font=font(15, bold=True))
    if sub:
        d.text((6, im.height + 22), sub, fill=NEUTRAL, font=font(13))
    return out


def grid(panels, cols, gap=10, bg=PANEL_BG):
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), bg)
    for i, p in enumerate(panels):
        canvas.paste(p, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return canvas


def load_clean():
    """-> {file_name: (width, height, [bbox, ...])} across all splits."""
    out = {}
    for sp in ("train", "val", "test"):
        d = json.load(open(f"{CCTV}/annotations/{sp}.json"))
        boxes = defaultdict(list)
        for a in d["annotations"]:
            boxes[a["image_id"]].append(a["bbox"])
        for i in d["images"]:
            out[i["file_name"]] = (i["width"], i["height"], boxes.get(i["id"], []))
    return out


# ---------------------------------------------------------------- figure 1
def fig_leakage():
    """The same photograph, re-augmented, sitting in train and in test."""
    audit = json.load(open("reports/dup_audit.json"))
    cluster_of = audit["assignments"]
    groups = defaultdict(list)
    stem_of = {}
    for split in ("train", "valid", "test"):
        d = json.load(open(f"{RAW_CCTV}/{split}/_annotations.coco.json"))
        for i in d["images"]:
            fn = i["file_name"]
            cid = cluster_of.get(fn)
            if cid is None:
                continue
            groups[cid].append((split, f"{RAW_CCTV}/{split}/{fn}"))
            stem_of[cid] = re.sub(r"\.rf\.[A-Za-z0-9]+\.jpg$", "", fn)
    cand = [(c, v) for c, v in groups.items()
            if {"train", "test"} <= {s for s, _ in v} and len(v) >= 6]
    cid, copies = sorted(cand, key=lambda kv: -len(kv[1]))[0]
    orig = stem_of[cid]
    order = {"train": 0, "valid": 1, "test": 2}
    copies = sorted(copies, key=lambda c: order[c[0]])
    picks = copies[:3] + copies[-3:] if len(copies) > 6 else copies
    panels = []
    for split, path in picks:
        im = Image.open(path).convert("RGB").resize((260, 260))
        col = BAD if split == "test" else (GOOD if split == "train" else (237, 161, 0))
        ImageDraw.Draw(im).rectangle([0, 0, 259, 259], outline=col, width=5)
        panels.append(caption(im, split.upper(), None, col))
    g = grid(panels, 3)
    out = Image.new("RGB", (g.width, g.height + 62), PANEL_BG)
    d = ImageDraw.Draw(out)
    d.text((4, 4), "Satu foto yang sama, muncul di train DAN di test",
           fill=INK, font=font(19, bold=True))
    d.text((4, 28), f"{orig[:56]}  -  {len(copies)} salinan augmentasi dari satu foto",
           fill=NEUTRAL, font=font(14))
    out.paste(g, (0, 58))
    out.save(f"{FIG}/fig_leakage.png")
    print(f"fig_leakage.png  ({orig}, {len(copies)} copies)")


# ---------------------------------------------------------------- figure 2
def fig_domain(clean):
    """What in this dataset matches an indoor-store camera, and what is noise."""
    T = 240

    def row(paths, border, boxes_of=None):
        panels = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            s = T / im.width
            im = im.resize((T, int(im.height * s)))
            if boxes_of is not None:
                draw_boxes(im, boxes_of(os.path.basename(p)), width=2, scale=s)
            ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1],
                                         outline=border, width=3)
            panels.append(im)
        return grid(panels, len(panels))

    # indoor families, preferring frames with several people
    indoor = [f for f in clean if family_of(f)[1] == "indoor"]
    indoor.sort(key=lambda f: (-min(len(clean[f][2]), 6), f))
    good_files = [f"{CCTV}/images/{indoor[i]}" for i in (0, 60, 130, 210, 300)]

    # the youtube family is dropped during cleaning, so these no longer exist in
    # the dataset -- pull them from the raw export to show what was excluded
    noise = sorted(p.replace("\\", "/") for p in
                   __import__("glob").glob(f"{RAW_CCTV}/*/youtube*.jpg"))
    noise_files = [noise[i] for i in (0, 12, 30, 55, 90)] if len(noise) > 90 else noise[:5]

    r1 = row(good_files, GOOD, boxes_of=lambda fn: clean[fn][2])
    r2 = row(noise_files, BAD)
    W = max(r1.width, r2.width)
    out = Image.new("RGB", (W, r1.height + r2.height + 112), PANEL_BG)
    d = ImageDraw.Draw(out)
    d.text((4, 6), "Paling relevan untuk CCTV toko indoor",
           fill=GOOD, font=font(18, bold=True))
    d.text((4, 28), "kamera plafon, sudut 30-45 derajat, rak dan perabot menutupi orang",
           fill=NEUTRAL, font=font(13))
    out.paste(r1, (0, 50))
    y = 50 + r1.height + 14
    d.text((4, y), "Bukan CCTV sama sekali - 140 foto, sudah dibuang dari dataset",
           fill=BAD, font=font(18, bold=True))
    d.text((4, y + 22), "webcam dan selfie dari YouTube: sudut, jarak dan lensa tidak ada hubungannya",
           fill=NEUTRAL, font=font(13))
    out.paste(r2, (0, y + 44))
    out = out.crop((0, 0, W, y + 52 + r2.height))
    out.save(f"{FIG}/fig_domain.png")
    print("fig_domain.png")


# ---------------------------------------------------------------- stats
def stats(clean):
    out = {}
    man = list(csv.DictReader(open(f"{CCTV}/manifest.csv")))
    audit = json.load(open("reports/dup_audit.json"))

    counts = [len(v[2]) for v in clean.values()]
    heights = [b[3] for v in clean.values() for b in v[2]]
    out["images"] = len(clean)
    out["boxes"] = len(heights)
    out["density_mean"] = float(np.mean(counts))
    out["density_hist"] = np.histogram(np.clip(counts, 0, 30), bins=15, range=(0, 30))[0].tolist()
    out["empty_images"] = int(sum(1 for c in counts if c == 0))
    out["boxh_hist"] = np.histogram(np.clip(np.array(heights) / 640, 0, 1),
                                    bins=20, range=(0, 1))[0].tolist()
    out["boxh_median_px"] = float(np.median(heights))

    out["splits"] = dict(Counter(r["split"] for r in man))
    out["boxes_per_split"] = {
        sp: sum(len(clean[r["file"]][2]) for r in man if r["split"] == sp)
        for sp in ("train", "val", "test")}
    out["scenes"] = len(set(r["scene"] for r in man))
    out["copies_hist"] = {str(k): v for k, v in sorted(
        Counter(min(int(r["copies_in_export"]), 10) for r in man).items())}

    out["audit"] = {k: audit[k] for k in
                    ("files", "stems", "stems_multi_photo", "distinct_photos",
                     "photos_with_copies", "photos_leaking", "files_in_leaking_photos",
                     "threshold")}

    fam = Counter()
    kind = Counter()
    for r in man:
        label, k = family_of(r["file"])
        fam[label] += 1
        kind[k] += 1
    out["families"] = dict(fam.most_common())
    out["domain_kind"] = dict(kind.most_common())

    json.dump(out, open("reports/stats.json", "w"), indent=1)
    print("stats.json")
    print("  domain:", dict(kind.most_common()))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    clean = load_clean()
    fig_leakage()
    fig_domain(clean)
    stats(clean)
