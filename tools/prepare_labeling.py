"""Sample frames from your own CCTV for labelling, and pre-annotate them.

Labelling is the expensive step, so this script spends effort making each
labelled frame count:

  1. SPREAD IN TIME. Never two frames from the same moment. The Roboflow
     dataset this project started from failed exactly here -- near-identical
     frames landed in train and test and made its mAP meaningless.
  2. DROP FRAMES WHERE NOTHING MOVED. A fixed CCTV pointed at a quiet shop
     produces thousands of frames that differ only in sensor noise.
     NOTE: this does NOT reuse the HSV descriptor from audit_duplicates.py.
     That one separates different SCENES; on a fixed camera the histogram is
     dominated by the unchanging background, so people moving barely shift it
     and it discards almost everything (measured: 88 of 99 candidates dropped,
     leaving 11 frames out of a requested 60). Instead this compares downscaled
     grayscale pixels directly, which is what actually registers a person
     moving through the frame.
  3. STRATIFY BY CROWDING. A store is empty most of the time, so uniform
     sampling yields mostly empty frames -- and crowded frames are exactly where
     a counter fails. Busy frames are deliberately over-sampled.
  4. PRE-ANNOTATE. Draft boxes at a deliberately LOW threshold, so correcting
     is mostly deleting rather than drawing.

    python tools/prepare_labeling.py --source toko.mp4 --n 300
    python tools/prepare_labeling.py --source toko.mp4 --n 300 --no-predraw

Output goes to data/store/<name>/ as images + YOLO labels + classes.txt, which
CVAT, Label Studio, LabelImg and Roboflow all import directly.

READ THIS BEFORE LABELLING
--------------------------
The draft boxes come from a model that is KNOWN to miss seated, crouching and
table-occluded people -- that is the measured gap this dataset exists to close.
If you only delete wrong boxes and never add missing ones, you teach the model
exactly the blind spot it already has. Every seated or half-hidden person needs
a box, including where the draft shows nothing.
"""
import argparse
import csv
import os
import sys

os.environ["YOLO_AUTOINSTALL"] = "False"

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(ROOT, "data", "store")
WEIGHTS = os.path.join(ROOT, "runs", "person", "default", "weights", "best.pt")
DRAFT_CONF = 0.15     # below the 0.25 operating point: favour recall while drafting
# Mean absolute pixel difference (0-255) below which two frames count as the
# same moment. Tune per camera with --min-change: a busy shop tolerates a
# higher value, a near-static stockroom needs a lower one.
MIN_CHANGE = 2.0

# A store is quiet most of the time. Uniform sampling would spend the whole
# labelling budget on empty frames, so each crowd bucket gets a fixed share.
SHARE = {"0": 0.15, "1-2": 0.30, "3-5": 0.30, "6+": 0.25}


def bucket_of(n):
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"


def thumb(bgr):
    """Small grayscale thumbnail; differences here mean something moved."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)


def changed_enough(t, kept_thumbs, thr):
    """True when this frame differs from every kept frame by more than thr."""
    if not kept_thumbs:
        return True
    return min(float(np.abs(t - k).mean()) for k in kept_thumbs) > thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=300, help="target jumlah frame")
    ap.add_argument("--min-gap-sec", type=float, default=2.0,
                    help="jarak minimum antar frame yang diambil")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--no-predraw", action="store_true",
                    help="jangan buat label draft; mulai dari kosong")
    ap.add_argument("--min-change", type=float, default=MIN_CHANGE,
                    help="selisih piksel rata-rata (0-255) minimum antar frame; "
                         "naikkan kalau terlalu banyak frame mirip lolos")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    src = os.path.abspath(a.source).replace("\\", "/")
    if not os.path.exists(src):
        sys.exit(f"video tidak ada: {src}  (pakai path Windows D:/..., bukan /d/...)")
    name = a.name or os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"tidak bisa membuka video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{os.path.basename(src)}  {W}x{H}  {fps:.0f} fps  "
          f"{total} frame  {total / max(fps, 1) / 60:.1f} menit")

    gap = max(1, int(a.min_gap_sec * fps))
    pool = list(range(0, total, gap))
    if len(pool) < a.n:
        print(f"PERINGATAN: video hanya menyediakan {len(pool)} frame pada jarak "
              f"{a.min_gap_sec}s. Turunkan --min-gap-sec atau rekam lebih lama.")
    print(f"kandidat: {len(pool)} frame (jarak >= {a.min_gap_sec}s)")

    from ultralytics import YOLO
    model = None if a.no_predraw else YOLO(a.weights)

    kept, thumbs, dropped = [], [], 0
    for k, fi in enumerate(pool):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        t = thumb(img)
        if not changed_enough(t, thumbs, a.min_change):
            dropped += 1
            continue
        boxes = []
        if model is not None:
            r = model.predict(img, conf=DRAFT_CONF, imgsz=640, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
                    boxes.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                  (x2 - x1) / W, (y2 - y1) / H))
        thumbs.append(t)
        kept.append((fi, img, boxes))
        if k and k % 200 == 0:
            print(f"  scan {k}/{len(pool)}  disimpan {len(kept)}  "
                  f"duplikat dibuang {dropped}")
    cap.release()
    if not kept:
        sys.exit("tidak ada frame yang bisa diambil")
    print(f"kandidat unik: {len(kept)}  (nyaris tak berubah, dibuang: {dropped})")
    if len(kept) < a.n * 0.5:
        print(f"CATATAN: hanya {len(kept)} frame lolos dari {a.n} yang diminta. "
              f"Turunkan --min-change (sekarang {a.min_change}) atau rekam lebih lama.")

    by_bucket = {b: [] for b in SHARE}
    for rec in kept:
        by_bucket[bucket_of(len(rec[2]))].append(rec)
    chosen, taken = [], set()
    for b, frac in SHARE.items():
        want, have = int(a.n * frac), by_bucket[b]
        if not have:
            continue
        step = max(1, len(have) // max(want, 1))
        for rec in have[::step][:want]:
            chosen.append(rec); taken.add(rec[0])
    if len(chosen) < a.n:   # a thin bucket: top up from whatever is left
        rest = [r for r in kept if r[0] not in taken]
        step = max(1, len(rest) // max(a.n - len(chosen), 1))
        chosen.extend(rest[::step][:a.n - len(chosen)])
    chosen.sort(key=lambda r: r[0])

    rows = []
    for fi, img, boxes in chosen:
        stem = f"{name}_{fi:07d}"
        cv2.imwrite(f"{out}/images/{stem}.jpg", img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(f"{out}/labels/{stem}.txt", "w") as f:
            f.write("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                              for cx, cy, w, h in boxes))
        rows.append({"file": f"{stem}.jpg", "frame": fi,
                     "detik": round(fi / max(fps, 1), 1), "draft_boxes": len(boxes)})

    with open(f"{out}/classes.txt", "w") as f:
        f.write("person\n")
    with open(f"{out}/manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    hist = {}
    for r in rows:
        hist[bucket_of(r["draft_boxes"])] = hist.get(bucket_of(r["draft_boxes"]), 0) + 1
    print(f"\n{len(rows)} frame disimpan -> {out}")
    print("sebaran kotak draft:", {b: hist.get(b, 0) for b in SHARE})
    print(f"total kotak draft  : {sum(r['draft_boxes'] for r in rows)}")
    print("\nINGAT: kotak draft berasal dari model yang TERBUKTI melewatkan orang")
    print("duduk dan tertutup meja. Menghapus kotak yang salah saja tidak cukup --")
    print("orang yang terlewat WAJIB ditambahkan, karena justru itu celahnya.")
    print(f"\nlangkah berikut: label di CVAT/Label Studio, lalu")
    print(f"  python tools/merge_store_data.py --store {out}")


if __name__ == "__main__":
    main()
