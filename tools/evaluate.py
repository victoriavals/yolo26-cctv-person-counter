"""Evaluate a person detector on the CCTV-person test split.

Answers two different questions, because they have different answers:

  1. Detection quality -- mAP, via pycocotools against the COCO ground truth.
     Uses the reference implementation rather than a hand-rolled AP, and gets
     the small/medium/large breakdown for free.

  2. Counting quality -- MAE, RMSE, bias and within-1 accuracy of the number of
     people per frame. This is the metric the product is judged on, and the
     conf threshold that minimises it is usually NOT the one that maximises F1.

Everything is broken down by domain kind (indoor is the deployment domain) and
by crowd density, because the aggregate hides both.

    python tools/evaluate.py baseline
    python tools/evaluate.py --all
    python tools/evaluate.py tuned --split val

Baseline is the pretrained COCO yolo26s restricted to class `person`. Nothing
here is meaningful without it: if a fine-tune cannot beat the off-the-shelf
model, that is the finding.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CCTV = os.path.join(ROOT, "data", "cctv-person")
YOLO_DIR = os.path.join(CCTV, "yolo").replace("\\", "/")
OUT_DIR = os.path.join(ROOT, "reports", "eval")

TARGETS = {
    "baseline": os.path.join(ROOT, "models", "yolo26s.pt"),
    "default": os.path.join(ROOT, "runs", "person", "default", "weights", "best.pt"),
    "tuned": os.path.join(ROOT, "runs", "person", "tuned", "weights", "best.pt"),
    "indoor": os.path.join(ROOT, "runs", "person", "indoor", "weights", "best.pt"),
}

# Low enough that the threshold sweep has something to sweep; raising it would
# silently truncate the PR curve.
PRED_CONF = 0.001
PRED_CHUNK = 16     # bounds VRAM; see the note in predict()
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)
MATCH_IOU = 0.5


# ----------------------------------------------------------------- data
def load_gt(split):
    """-> (coco_dict, {file_name: image_id}, {image_id: [xywh, ...]})"""
    d = json.load(open(f"{CCTV}/annotations/{split}.json"))
    id_of = {i["file_name"]: i["id"] for i in d["images"]}
    boxes = {i["id"]: [] for i in d["images"]}
    for a in d["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    return d, id_of, boxes


def load_meta():
    """file_name -> (kind, density_bucket) from the cleaning manifest."""
    return {r["file"]: (r["kind"], r["density_bucket"])
            for r in csv.DictReader(open(f"{CCTV}/manifest.csv"))}


def density_of(n):
    """Bucket by the ACTUAL count in this frame, not the scene median."""
    return "0-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "10+"


# ----------------------------------------------------------------- predict
def predict(weights, split, imgsz=640):
    """-> {file_name: ndarray[N,5] of xywh+score}, all boxes above PRED_CONF."""
    from ultralytics import YOLO

    model = YOLO(weights)
    # the pretrained checkpoint is 80-class COCO; class 0 is `person`. A
    # fine-tuned checkpoint has a single class, already 0.
    n_classes = len(model.names)
    paths = sorted(
        f"{YOLO_DIR}/images/{split}/{f}"
        for f in os.listdir(f"{YOLO_DIR}/images/{split}") if f.endswith(".jpg"))
    print(f"  {os.path.basename(weights)}: {n_classes} kelas, {len(paths)} gambar")

    # Two reasons for chunking instead of handing predict() the whole list:
    #  1. Given a list source, ultralytics renames results to image0.jpg,
    #     image1.jpg ... -- r.path is NOT the input path, so filenames have to
    #     come from the input order, not from the result.
    #  2. It also loads the entire list as one batch; all 475 test images at
    #     once asks the allocator for 6.2 GB and trips OOM on an 8 GB card.
    out = {}
    for i in range(0, len(paths), PRED_CHUNK):
        chunk = paths[i:i + PRED_CHUNK]
        results = model.predict(chunk, classes=[0], conf=PRED_CONF, imgsz=imgsz,
                                verbose=False)
        for path, r in zip(chunk, results):
            fn = os.path.basename(path)
            b = r.boxes
            if b is None or len(b) == 0:
                out[fn] = np.zeros((0, 5), dtype=np.float32)
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            xywh = np.stack([xyxy[:, 0], xyxy[:, 1],
                             xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
            out[fn] = np.concatenate([xywh, conf[:, None]], 1).astype(np.float32)
    return out, n_classes


# ----------------------------------------------------------------- matching
def iou_matrix(a, b):
    """a: [N,4] xywh, b: [M,4] xywh -> [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match(pred, gt, thr=MATCH_IOU):
    """Confidence-ordered greedy matching, COCO style.

    Each prediction takes the best GT still free -- not simply its global best,
    which would drop a valid match whenever the top choice was already taken and
    quietly understate recall.

    -> (tp, fp, fn, matched_pred_idx)
    """
    if len(pred) == 0:
        return 0, 0, len(gt), []
    if len(gt) == 0:
        return 0, len(pred), 0, []
    order = np.argsort(-pred[:, 4])
    M = iou_matrix(pred[order, :4], np.asarray(gt, dtype=np.float32))
    free = np.ones(M.shape[1], dtype=bool)
    matched = []
    for i in range(M.shape[0]):
        row = np.where(free, M[i], -1.0)
        j = int(np.argmax(row))
        if row[j] >= thr:
            free[j] = False
            matched.append(int(order[i]))
    tp = len(matched)
    return tp, len(pred) - tp, len(gt) - tp, matched


# ----------------------------------------------------------------- metrics
def coco_metrics(preds, gt_dict, id_of, img_subset=None):
    """mAP via pycocotools. -> dict, or None when there is nothing to score."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dets = []
    for fn, p in preds.items():
        iid = id_of.get(fn)
        if iid is None or (img_subset is not None and iid not in img_subset):
            continue
        for x, y, w, h, s in p:
            dets.append({"image_id": int(iid), "category_id": 1,
                         "bbox": [float(x), float(y), float(w), float(h)],
                         "score": float(s)})
    if not dets:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        tmp = os.path.join(OUT_DIR, "_gt_tmp.json")
        json.dump(gt_dict, open(tmp, "w"))
        cg = COCO(tmp)
        cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, "bbox")
        if img_subset is not None:
            E.params.imgIds = sorted(img_subset)
        E.evaluate(); E.accumulate(); E.summarize()
        os.remove(tmp)
    s = E.stats
    return {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
            "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
            "AR_100": float(s[8])}


def pr_at(preds, gt_boxes, id_of, conf, files):
    tp = fp = fn = 0
    for f in files:
        p = preds[f]
        p = p[p[:, 4] >= conf]
        a, b, c, _ = match(p, gt_boxes[id_of[f]])
        tp += a; fp += b; fn += c
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return {"P": P, "R": R, "F1": F1, "tp": tp, "fp": fp, "fn": fn}


def counting(preds, gt_boxes, id_of, conf, files):
    """The metric the product is actually judged on."""
    err = []
    for f in files:
        p = preds[f]
        n_pred = int((p[:, 4] >= conf).sum())
        n_gt = len(gt_boxes[id_of[f]])
        err.append((n_pred - n_gt, n_gt))
    e = np.array([x[0] for x in err], dtype=float)
    return {"MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()),
            "within1_pct": float((np.abs(e) <= 1).mean() * 100),
            "within2_pct": float((np.abs(e) <= 2).mean() * 100),
            "n_images": len(files)}


def evaluate(name, weights, split, imgsz=640, suffix=""):
    os.makedirs(OUT_DIR, exist_ok=True)   # coco_metrics writes a temp file here
    gt_dict, id_of, gt_boxes = load_gt(split)
    meta = load_meta()
    preds, n_classes = predict(weights, split, imgsz)
    files = sorted(f for f in preds if f in id_of)
    if not files:
        sys.exit(f"tidak ada prediksi yang cocok dengan ground truth "
                 f"({len(preds)} prediksi, {len(id_of)} gambar GT) -- "
                 f"cek pemetaan nama file")

    # threshold sweep -- pick the operating point that minimises counting error,
    # and record where max-F1 falls so the gap between them is visible
    sweep = []
    for c in SWEEP:
        pr = pr_at(preds, gt_boxes, id_of, c, files)
        ct = counting(preds, gt_boxes, id_of, c, files)
        sweep.append({"conf": float(c), **{k: pr[k] for k in ("P", "R", "F1")},
                      "MAE": ct["MAE"], "bias": ct["bias"]})
    best_count = min(sweep, key=lambda r: r["MAE"])["conf"]
    best_f1 = max(sweep, key=lambda r: r["F1"])["conf"]

    res = {
        "name": name, "weights": weights, "split": split, "imgsz": imgsz,
        "model_classes": n_classes, "images": len(files),
        "conf_best_counting": best_count, "conf_best_f1": best_f1,
        "overall": coco_metrics(preds, gt_dict, id_of),
        "at_best_counting": {
            **pr_at(preds, gt_boxes, id_of, best_count, files),
            **counting(preds, gt_boxes, id_of, best_count, files)},
        "at_best_f1": {
            **pr_at(preds, gt_boxes, id_of, best_f1, files),
            **counting(preds, gt_boxes, id_of, best_f1, files)},
        "sweep": sweep,
    }

    # --- breakdown by deployment domain
    res["by_kind"] = {}
    for kind in ("indoor", "outdoor", "unknown", "semi"):
        sub = [f for f in files if meta.get(f, ("unknown", ""))[0] == kind]
        if len(sub) < 20:
            continue
        ids = {id_of[f] for f in sub}
        res["by_kind"][kind] = {
            "n": len(sub),
            "coco": coco_metrics(preds, gt_dict, id_of, ids),
            **pr_at(preds, gt_boxes, id_of, best_count, sub),
            **counting(preds, gt_boxes, id_of, best_count, sub)}

    # --- breakdown by how crowded the frame actually is
    res["by_density"] = {}
    groups = {}
    for f in files:
        groups.setdefault(density_of(len(gt_boxes[id_of[f]])), []).append(f)
    for b in ("0-2", "3-5", "6-10", "10+"):
        sub = groups.get(b, [])
        if len(sub) < 10:
            continue
        ids = {id_of[f] for f in sub}
        res["by_density"][b] = {
            "n": len(sub),
            "coco": coco_metrics(preds, gt_dict, id_of, ids),
            **pr_at(preds, gt_boxes, id_of, best_count, sub),
            **counting(preds, gt_boxes, id_of, best_count, sub)}

    # --- per image, for the scatter plot and the error gallery
    res["per_image"] = []
    for f in files:
        p = preds[f][preds[f][:, 4] >= best_count]
        n_gt = len(gt_boxes[id_of[f]])
        tp, fp, fn, _ = match(p, gt_boxes[id_of[f]])
        res["per_image"].append({
            "file": f, "kind": meta.get(f, ("unknown", ""))[0],
            "gt": n_gt, "pred": int(len(p)), "tp": tp, "fp": fp, "fn": fn})

    json.dump(res, open(f"{OUT_DIR}/{name}{suffix}.json", "w"), indent=1)
    # raw boxes, so the figure script can draw predictions without paying for a
    # second inference pass over the split
    np.savez_compressed(f"{OUT_DIR}/{name}{suffix}_preds.npz",
                        **{f: preds[f] for f in files})
    report(res)
    return res


def report(r):
    o, c = r["overall"], r["at_best_counting"]
    print(f"\n{'=' * 62}\n{r['name']}  ({r['images']} gambar, split {r['split']})\n{'=' * 62}")
    print(f"  mAP50      {o['mAP50']:.4f}    mAP50-95 {o['mAP50_95']:.4f}")
    print(f"  AP small   {o['AP_small']:.4f}  medium {o['AP_medium']:.4f}  "
          f"large {o['AP_large']:.4f}")
    print(f"\n  conf terbaik untuk HITUNGAN : {r['conf_best_counting']}"
          f"   (untuk F1: {r['conf_best_f1']})")
    print(f"  MAE {c['MAE']:.3f}  RMSE {c['RMSE']:.3f}  bias {c['bias']:+.3f}  "
          f"dalam +-1 {c['within1_pct']:.1f}%")
    print(f"  P {c['P']:.4f}  R {c['R']:.4f}  F1 {c['F1']:.4f}")
    if r["by_kind"]:
        print("\n  per domain (pada conf hitungan terbaik):")
        for k, v in r["by_kind"].items():
            print(f"    {k:<9} n={v['n']:<4} mAP50={v['coco']['mAP50']:.4f}  "
                  f"MAE={v['MAE']:.3f}  R={v['R']:.4f}")
    if r["by_density"]:
        print("\n  per kepadatan:")
        for k, v in r["by_density"].items():
            print(f"    {k:<6} n={v['n']:<4} mAP50={v['coco']['mAP50']:.4f}  "
                  f"MAE={v['MAE']:.3f}  R={v['R']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; lower = faster on CPU, costs accuracy")
    a = ap.parse_args()

    names = list(TARGETS) if a.all else a.targets
    if not names:
        sys.exit(f"pilih salah satu: {list(TARGETS)}  (atau --all)")
    os.chdir(ROOT)
    for n in names:
        if n not in TARGETS:
            sys.exit(f"target tidak dikenal: {n}")
        if not os.path.exists(TARGETS[n]):
            print(f"lewati {n}: {TARGETS[n]} belum ada")
            continue
        sfx = "" if a.imgsz == 640 else f"_{a.imgsz}"
        evaluate(n, TARGETS[n], a.split, a.imgsz, sfx)


if __name__ == "__main__":
    main()
