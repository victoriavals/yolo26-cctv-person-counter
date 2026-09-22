"""Build the evaluation figures from whatever runs exist in reports/eval/.

Eight views, each answering a question the aggregate mAP cannot:

  curves.png        did the runs converge, and did any overfit
  sweep.png         MAE and F1 against conf -- they peak at different thresholds
  scatter.png       predicted vs actual count per frame, coloured by domain
  breakdown.png     mAP50 and MAE per domain and per crowd density
  reliability.png   does a confidence of 0.8 actually mean 80% correct
  grid_<run>.png    ground truth vs prediction on the best and worst frames
  errors_<run>.png  the confident false positives and the large misses

Reads reports/eval/<run>.json and <run>_preds.npz, so nothing re-runs inference.

    python tools/make_eval_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import MATCH_IOU, iou_matrix, match  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL = os.path.join(ROOT, "reports", "eval")
FIG = os.path.join(EVAL, "figures")
IMGS = os.path.join(ROOT, "data", "cctv-person", "yolo", "images")
RUNS = os.path.join(ROOT, "runs", "person")

ORDER = ["baseline", "default", "tuned", "indoor"]
COLOR = {"baseline": "#77838f", "default": "#eb6834",
         "tuned": "#2a78d6", "indoor": "#12805a"}
GT_C, TP_C, FP_C, FN_C = (60, 200, 255), (40, 200, 120), (235, 80, 70), (250, 190, 40)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def load_runs():
    out = {}
    for n in ORDER:
        p = f"{EVAL}/{n}.json"
        if os.path.exists(p):
            out[n] = json.load(open(p))
    if not out:
        sys.exit(f"tidak ada hasil evaluasi di {EVAL} -- jalankan tools/evaluate.py dulu")
    return out


def preds_of(name):
    p = f"{EVAL}/{name}_preds.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def font(sz):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", sz)
    except OSError:
        return ImageFont.load_default(sz)


# ------------------------------------------------------------------ 1 curves
def fig_curves():
    import csv
    found = [(n, f"{RUNS}/{n}/results.csv") for n in ORDER
             if os.path.exists(f"{RUNS}/{n}/results.csv")]
    if not found:
        print("curves: belum ada results.csv, dilewati")
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for n, p in found:
        rows = list(csv.DictReader(open(p)))
        k = {c.strip(): c for c in rows[0]}
        ep = [float(r[k["epoch"]]) for r in rows]
        get = lambda key: [float(r[k[key]]) for r in rows if key in k]  # noqa: E731
        for i, (key, title) in enumerate((
                ("train/box_loss", "box loss (train)"),
                ("metrics/mAP50(B)", "mAP50 (val)"),
                ("metrics/mAP50-95(B)", "mAP50-95 (val)"))):
            if key in k:
                ax[i].plot(ep, get(key), label=n, color=COLOR.get(n), lw=1.6)
                ax[i].set_title(title)
                ax[i].set_xlabel("epoch")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/curves.png", dpi=140); plt.close(fig)
    print("curves.png")


# ------------------------------------------------------------------ 2 sweep
def fig_sweep(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, r in runs.items():
        c = [s["conf"] for s in r["sweep"]]
        ax[0].plot(c, [s["MAE"] for s in r["sweep"]], color=COLOR.get(n), lw=1.8, label=n)
        ax[0].scatter([r["conf_best_counting"]],
                      [min(s["MAE"] for s in r["sweep"])],
                      color=COLOR.get(n), zorder=5, s=34, edgecolor="white", linewidth=1.2)
        ax[1].plot(c, [s["F1"] for s in r["sweep"]], color=COLOR.get(n), lw=1.8, label=n)
    ax[0].set_title("Galat hitung (MAE) vs threshold — titik = optimum")
    ax[0].set_xlabel("conf"); ax[0].set_ylabel("MAE orang/frame")
    ax[1].set_title("F1 vs threshold")
    ax[1].set_xlabel("conf"); ax[1].set_ylabel("F1")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/sweep.png", dpi=140); plt.close(fig)
    print("sweep.png")


# ------------------------------------------------------------------ 3 scatter
def fig_scatter(runs):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4), squeeze=False)
    kinds = {"indoor": "#2a78d6", "outdoor": "#eb6834",
             "unknown": "#9aa4ae", "semi": "#12805a"}
    for ax, (name, r) in zip(axes[0], runs.items()):
        pi = r["per_image"]
        hi = max(max(p["gt"] for p in pi), max(p["pred"] for p in pi)) + 1
        for k, col in kinds.items():
            xs = [p["gt"] for p in pi if p["kind"] == k]
            ys = [p["pred"] for p in pi if p["kind"] == k]
            if xs:
                ax.scatter(xs, ys, s=13, alpha=.5, color=col, label=k, linewidths=0)
        ax.plot([0, hi], [0, hi], color="#444", lw=1, ls="--")
        c = r["at_best_counting"]
        ax.set_title(f"{name}\nMAE {c['MAE']:.2f}  bias {c['bias']:+.2f}  "
                     f"±1 {c['within1_pct']:.0f}%")
        ax.set_xlabel("jumlah orang sebenarnya"); ax.set_ylabel("prediksi")
        ax.set_xlim(-.5, hi); ax.set_ylim(-.5, hi)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/scatter.png", dpi=140); plt.close(fig)
    print("scatter.png")


# ------------------------------------------------------------------ 4 breakdown
def fig_breakdown(runs):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))
    for col, (group, title) in enumerate((("by_kind", "domain"),
                                          ("by_density", "kepadatan (orang/frame)"))):
        keys = sorted({k for r in runs.values() for k in r.get(group, {})},
                      key=lambda k: (["0-2", "3-5", "6-10", "10+"].index(k)
                                     if k in ("0-2", "3-5", "6-10", "10+") else 0))
        if not keys:
            continue
        x = np.arange(len(keys))
        w = .8 / max(len(runs), 1)
        for i, (name, r) in enumerate(runs.items()):
            g = r.get(group, {})
            ax[0][col].bar(x + i * w, [g.get(k, {}).get("coco", {}).get("mAP50", 0)
                                       for k in keys], w, color=COLOR.get(name), label=name)
            ax[1][col].bar(x + i * w, [g.get(k, {}).get("MAE", 0) for k in keys],
                           w, color=COLOR.get(name))
        for row, lab in ((0, "mAP50"), (1, "MAE hitungan")):
            ax[row][col].set_xticks(x + .4 - w / 2)
            ax[row][col].set_xticklabels(
                [f"{k}\n(n={next((r[group][k]['n'] for r in runs.values() if k in r.get(group, {})), 0)})"
                 for k in keys], fontsize=8)
            ax[row][col].set_ylabel(lab)
            ax[row][col].set_title(f"{lab} per {title}")
    ax[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/breakdown.png", dpi=140); plt.close(fig)
    print("breakdown.png")


# ------------------------------------------------------------------ 5 reliability
def fig_reliability(runs):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    gt = json.load(open(os.path.join(ROOT, "data/cctv-person/annotations/test.json")))
    boxes = {i["file_name"]: [] for i in gt["images"]}
    byid = {i["id"]: i["file_name"] for i in gt["images"]}
    for a in gt["annotations"]:
        boxes[byid[a["image_id"]]].append(a["bbox"])

    edges = np.linspace(0, 1, 11)
    for name in runs:
        P = preds_of(name)
        if P is None:
            continue
        conf_all, correct_all = [], []
        for f, p in P.items():
            if len(p) == 0:
                continue
            g = np.asarray(boxes.get(f, []), dtype=np.float32)
            ok = np.zeros(len(p), dtype=bool)
            if len(g):
                _, _, _, m = match(p, g)
                ok[m] = True
            conf_all.append(p[:, 4]); correct_all.append(ok)
        if not conf_all:
            continue
        c = np.concatenate(conf_all); ok = np.concatenate(correct_all)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (c >= lo) & (c < hi)
            if m.sum() >= 30:
                xs.append((lo + hi) / 2); ys.append(ok[m].mean())
        ax.plot(xs, ys, "o-", color=COLOR.get(name), lw=1.6, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#444", lw=1)
    ax.set_xlabel("confidence"); ax.set_ylabel(f"proporsi benar (IoU>={MATCH_IOU})")
    ax.set_title("Kalibrasi: apakah conf 0,8 berarti 80% benar?")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/reliability.png", dpi=140); plt.close(fig)
    print("reliability.png")


# ------------------------------------------------------------------ 6/7 images
def draw(fn, pred, gt, conf):
    im = Image.open(f"{IMGS}/test/{fn}").convert("RGB")
    d = ImageDraw.Draw(im)
    p = pred[pred[:, 4] >= conf]
    g = np.asarray(gt, dtype=np.float32)
    _, _, _, m = match(p, g)
    hit = set(m)
    for x, y, w, h in g:
        d.rectangle([x, y, x + w, y + h], outline=GT_C, width=2)
    for i, (x, y, w, h, s) in enumerate(p):
        col = TP_C if i in hit else FP_C
        d.rectangle([x, y, x + w, y + h], outline=col, width=2)
    return im, len(g), len(p)


def panel(im, caption, sub, color=(245, 245, 242)):
    out = Image.new("RGB", (im.width, im.height + 36), (22, 22, 21))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((5, im.height + 3), caption, fill=color, font=font(13))
    d.text((5, im.height + 19), sub, fill=(150, 150, 145), font=font(12))
    return out


def grid(panels, cols):
    w = max(p.width for p in panels); h = max(p.height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    c = Image.new("RGB", (cols * w + (cols - 1) * 8, rows * h + (rows - 1) * 8), (22, 22, 21))
    for i, p in enumerate(panels):
        c.paste(p, ((i % cols) * (w + 8), (i // cols) * (h + 8)))
    return c


def fig_grids(runs):
    gt = json.load(open(os.path.join(ROOT, "data/cctv-person/annotations/test.json")))
    byid = {i["id"]: i["file_name"] for i in gt["images"]}
    boxes = {i["file_name"]: [] for i in gt["images"]}
    for a in gt["annotations"]:
        boxes[byid[a["image_id"]]].append(a["bbox"])

    for name, r in runs.items():
        P = preds_of(name)
        if P is None:
            continue
        conf = r["conf_best_counting"]
        # rank by absolute counting error, only frames with people in them
        pi = [p for p in r["per_image"] if p["gt"] >= 2]
        pi.sort(key=lambda p: abs(p["pred"] - p["gt"]))
        for tag, sel in (("baik", pi[:6]), ("buruk", pi[-6:][::-1])):
            panels = []
            for p in sel:
                im, ng, npd = draw(p["file"], P[p["file"]], boxes[p["file"]], conf)
                im.thumbnail((330, 330))
                panels.append(panel(im, f"GT {ng}  ->  prediksi {npd}",
                                    f"{p['kind']}  tp{p['tp']} fp{p['fp']} fn{p['fn']}",
                                    (120, 230, 170) if tag == "baik" else (240, 110, 100)))
            g = grid(panels, 3)
            out = Image.new("RGB", (g.width, g.height + 30), (22, 22, 21))
            ImageDraw.Draw(out).text(
                (4, 6), f"{name} — 6 frame {tag} (biru=GT, hijau=benar, merah=salah), conf={conf}",
                fill=(245, 245, 242), font=font(15))
            out.paste(g, (0, 28))
            out.save(f"{FIG}/grid_{name}_{tag}.png")
        print(f"grid_{name}_*.png")


def fig_errors(runs):
    gt = json.load(open(os.path.join(ROOT, "data/cctv-person/annotations/test.json")))
    byid = {i["id"]: i["file_name"] for i in gt["images"]}
    boxes = {i["file_name"]: [] for i in gt["images"]}
    for a in gt["annotations"]:
        boxes[byid[a["image_id"]]].append(a["bbox"])

    for name, r in runs.items():
        P = preds_of(name)
        if P is None:
            continue
        conf = r["conf_best_counting"]
        fps, fns = [], []
        for f, p in P.items():
            p = p[p[:, 4] >= conf]
            g = np.asarray(boxes.get(f, []), dtype=np.float32)
            _, _, _, m = match(p, g)
            hit = set(m)
            for i, b in enumerate(p):
                if i not in hit:
                    fps.append((float(b[4]), f, b[:4]))
            if len(g):
                M = iou_matrix(g, p[:, :4]) if len(p) else np.zeros((len(g), 0))
                best = M.max(1) if M.size else np.zeros(len(g))
                for j, b in enumerate(g):
                    if best[j] < MATCH_IOU:
                        fns.append((float(b[2] * b[3]), f, b))
        fps.sort(key=lambda t: -t[0]); fns.sort(key=lambda t: -t[0])

        panels = []
        for score, f, b in fps[:6]:
            im = Image.open(f"{IMGS}/test/{f}").convert("RGB")
            ImageDraw.Draw(im).rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                         outline=FP_C, width=3)
            im.thumbnail((300, 300))
            panels.append(panel(im, f"FP conf {score:.2f}", f[:30], (240, 110, 100)))
        for area, f, b in fns[:6]:
            im = Image.open(f"{IMGS}/test/{f}").convert("RGB")
            ImageDraw.Draw(im).rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                         outline=FN_C, width=3)
            im.thumbnail((300, 300))
            panels.append(panel(im, f"FN luas {int(area)}px", f[:30], (250, 200, 90)))
        if not panels:
            continue
        g = grid(panels, 6)
        out = Image.new("RGB", (g.width, g.height + 30), (22, 22, 21))
        ImageDraw.Draw(out).text(
            (4, 6), f"{name} — atas: FP paling yakin. bawah: orang terbesar yang terlewat",
            fill=(245, 245, 242), font=font(15))
        out.paste(g, (0, 28))
        out.save(f"{FIG}/errors_{name}.png")
        print(f"errors_{name}.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load_runs()
    print("run ditemukan:", ", ".join(runs))
    fig_curves()
    fig_sweep(runs)
    fig_scatter(runs)
    fig_breakdown(runs)
    fig_reliability(runs)
    fig_grids(runs)
    fig_errors(runs)


if __name__ == "__main__":
    main()
