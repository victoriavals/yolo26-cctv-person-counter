"""Export the chosen model to ONNX, prove the export is faithful, and time it.

Three things, in order, because each only matters if the previous one held:

  1. Export best.pt -> ONNX.
  2. Parity: run PyTorch and ONNX over the same images and compare detections.
     An export that silently changes the output is worse than no export.
  3. Speed: PyTorch on GPU, PyTorch on CPU, ONNX on CPU. The store box will
     probably not have a GPU, so the CPU numbers are the ones that decide
     whether this is deployable.

    python tools/export.py
    python tools/export.py --run tuned --n 60
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# Must be set BEFORE ultralytics is imported. Loading an .onnx model while CUDA
# is present makes ultralytics try to `pip install onnxruntime-gpu`. On this
# system-wide Python install that fails halfway: the Python files get replaced
# but the native DLL rename is denied, leaving onnxruntime broken with
# "module 'onnxruntime.capi._pybind_state' has no attribute 'OrtCompileApiFlags'".
# Recovering needs a manual `pip install --force-reinstall onnxruntime==1.21.1`.
os.environ["YOLO_AUTOINSTALL"] = "False"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMGS = os.path.join(ROOT, "data", "cctv-person", "yolo", "images", "test")
OUT = os.path.join(ROOT, "reports", "eval")
CONF = 0.25          # the operating point chosen by tools/evaluate.py
PARITY_IOU = 0.95    # boxes this close are the same detection


def sample_images(n):
    fs = sorted(f for f in os.listdir(IMGS) if f.endswith(".jpg"))
    step = max(1, len(fs) // n)
    return [os.path.join(IMGS, f) for f in fs[::step][:n]]


def detect(model, paths, device=None, chunk_size=8):
    """-> list of [N,5] arrays (xyxy + conf), one per image, in input order.

    chunk_size must be 1 for the ONNX model: the export fixes the batch
    dimension at 1, and feeding it more raises
    "Got invalid dimensions for input: images". Batch 1 is also how the model
    will actually run in a store, one frame at a time.
    """
    out = []
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
        kw = {"device": device} if device is not None else {}
        for r in model.predict(chunk, conf=CONF, imgsz=640, verbose=False, **kw):
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 5), np.float32))
            else:
                out.append(np.concatenate(
                    [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1))
    return out


def iou_pairs(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def parity(pt, on):
    """How many detections agree between the two runtimes."""
    same = tot_pt = tot_on = 0
    dconf = []
    for a, b in zip(pt, on):
        tot_pt += len(a); tot_on += len(b)
        if len(a) == 0 or len(b) == 0:
            continue
        M = iou_pairs(a[:, :4], b[:, :4])
        free = np.ones(M.shape[1], bool)
        for i in range(M.shape[0]):
            row = np.where(free, M[i], -1)
            j = int(np.argmax(row))
            if row[j] >= PARITY_IOU:
                free[j] = False; same += 1
                dconf.append(abs(float(a[i, 4] - b[j, 4])))
    return {"pytorch_boxes": tot_pt, "onnx_boxes": tot_on, "matched": same,
            "match_pct": 100 * same / max(tot_pt, 1),
            "max_conf_delta": float(max(dconf)) if dconf else 0.0,
            "count_equal_images": int(sum(len(a) == len(b) for a, b in zip(pt, on))),
            "images": len(pt)}


def bench(model, paths, device, warm=5):
    for p in paths[:warm]:
        model.predict(p, conf=CONF, imgsz=640, verbose=False, device=device)
    t = time.perf_counter()
    for p in paths:
        model.predict(p, conf=CONF, imgsz=640, verbose=False, device=device)
    dt = time.perf_counter() - t
    return {"ms_per_frame": 1000 * dt / len(paths), "fps": len(paths) / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--n", type=int, default=40, help="gambar untuk parity & benchmark")
    a = ap.parse_args()
    os.chdir(ROOT)

    from ultralytics import YOLO
    import torch

    w = os.path.join(ROOT, "runs", "person", a.run, "weights", "best.pt")
    if not os.path.exists(w):
        sys.exit(f"bobot tidak ada: {w}")
    paths = sample_images(a.n)
    res = {"run": a.run, "weights": w, "conf": CONF, "n_images": len(paths)}

    print(f"=== 1. export ONNX ({a.run}) ===")
    onnx_path = YOLO(w).export(format="onnx", imgsz=640, opset=12, simplify=True)
    res["onnx"] = str(onnx_path)
    res["onnx_mb"] = round(os.path.getsize(onnx_path) / 1e6, 1)
    print(f"  -> {onnx_path}  ({res['onnx_mb']} MB)")

    print(f"\n=== 2. parity PyTorch vs ONNX ({len(paths)} gambar) ===")
    pt_dets = detect(YOLO(w), paths, device="cpu")
    on_dets = detect(YOLO(str(onnx_path), task="detect"), paths,
                     device="cpu", chunk_size=1)
    res["parity"] = parity(pt_dets, on_dets)
    p = res["parity"]
    print(f"  kotak PyTorch {p['pytorch_boxes']}  ONNX {p['onnx_boxes']}  "
          f"cocok {p['matched']} ({p['match_pct']:.1f}%)")
    print(f"  selisih conf maksimum {p['max_conf_delta']:.5f}")
    print(f"  gambar dengan jumlah deteksi identik: "
          f"{p['count_equal_images']}/{p['images']}")

    print(f"\n=== 3. kecepatan ({len(paths)} gambar, imgsz 640, conf {CONF}) ===")
    res["speed"] = {}
    m_pt = YOLO(w)
    if torch.cuda.is_available():
        res["speed"]["pytorch_gpu"] = bench(m_pt, paths, 0)
    res["speed"]["pytorch_cpu"] = bench(m_pt, paths, "cpu")
    res["speed"]["onnx_cpu"] = bench(
        YOLO(str(onnx_path), task="detect"), paths, "cpu")
    for k, v in res["speed"].items():
        print(f"  {k:<14}{v['ms_per_frame']:>8.1f} ms/frame   {v['fps']:>6.1f} FPS")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/export_{a.run}.json", "w"), indent=1)
    print(f"\ndisimpan -> reports/eval/export_{a.run}.json")


if __name__ == "__main__":
    main()
