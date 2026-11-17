"""Compare every inference backend on this machine, CPU and GPU, fairly.

Speed alone would mislead, so each backend is checked for two things:

  parity -- does it produce the SAME detections as the PyTorch CPU reference?
            A backend that is fast because it quietly drops boxes is not fast.
  speed  -- ms per frame at each inference size, after warm-up, one frame per
            call (how it runs in production, and the only mode fixed-batch
            exports accept).

Device targeting is ALWAYS explicit. Ultralytics falls back to OpenVINO's
`AUTO` device whenever anything besides CPU is enumerated, so a run meant to
measure the CPU can silently be dispatched to the GPU. `intel:cpu` /
`intel:gpu` pin it.

    python tools/benchmark_backends.py
    python tools/benchmark_backends.py --n 60 --sizes 640,416 --skip tensorrt_gpu
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

# Before ultralytics is imported: loading a non-PyTorch model with CUDA present
# makes it pip-install a GPU runtime, which half-succeeds on this system-wide
# Python and leaves the old one broken. See README.md.
os.environ["YOLO_AUTOINSTALL"] = "False"

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMGS = os.path.join(ROOT, "data", "cctv-person", "yolo", "images", "test")
OUT = os.path.join(ROOT, "reports", "eval")
STAGE = os.path.join(OUT, "backends")
CONF = 0.25
PARITY_IOU = 0.95

# label -> (export format, ultralytics device, hardware it actually runs on).
# format None means the .pt weights are used directly.
BACKENDS = [
    ("pytorch_gpu",  None,       0,           "GPU"),
    ("pytorch_cpu",  None,       "cpu",       "CPU"),
    ("onnx_cpu",     "onnx",     "cpu",       "CPU"),
    ("openvino_cpu", "openvino", "intel:cpu", "CPU"),
    ("openvino_gpu", "openvino", "intel:gpu", "GPU"),
    ("tensorrt_gpu", "engine",   0,           "GPU"),
]


def sample(n):
    fs = sorted(f for f in os.listdir(IMGS) if f.endswith(".jpg"))
    step = max(1, len(fs) // n)
    return [os.path.join(IMGS, f) for f in fs[::step][:n]]


def boxes_of(r):
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 5), np.float32)
    return np.concatenate([b.xyxy.cpu().numpy(),
                           b.conf.cpu().numpy()[:, None]], 1).astype(np.float32)


def detect(model, paths, device, imgsz, chunk):
    out = []
    for i in range(0, len(paths), chunk):
        for r in model.predict(paths[i:i + chunk], conf=CONF, imgsz=imgsz,
                               verbose=False, device=device):
            out.append(boxes_of(r))
    return out


def bench(model, paths, device, imgsz, warm=5):
    for p in paths[:warm]:
        model.predict(p, conf=CONF, imgsz=imgsz, verbose=False, device=device)
    t = time.perf_counter()
    for p in paths:
        model.predict(p, conf=CONF, imgsz=imgsz, verbose=False, device=device)
    return 1000 * (time.perf_counter() - t) / len(paths)


def iou(a, b):
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


def parity(ref, other):
    same = n_ref = n_oth = 0
    dconf = []
    for a, b in zip(ref, other):
        n_ref += len(a); n_oth += len(b)
        if len(a) == 0 or len(b) == 0:
            continue
        M = iou(a[:, :4], b[:, :4])
        free = np.ones(M.shape[1], bool)
        for i in range(M.shape[0]):
            row = np.where(free, M[i], -1)
            j = int(np.argmax(row))
            if row[j] >= PARITY_IOU:
                free[j] = False; same += 1
                dconf.append(abs(float(a[i, 4] - b[j, 4])))
    return {"ref_boxes": n_ref, "boxes": n_oth, "matched": same,
            "match_pct": 100 * same / max(n_ref, 1),
            "max_conf_delta": float(max(dconf)) if dconf else 0.0,
            "count_equal_images": int(sum(len(a) == len(b)
                                          for a, b in zip(ref, other)))}


def export_one(YOLO, pt, fmt, size):
    """Stage each export in its own directory.

    ultralytics writes exports beside the weights, so exporting a second size
    overwrites the first -- and if that model is still loaded, the .bin or
    .engine is locked ("Can't open bin file"). An OpenVINO model is also
    identified by its DIRECTORY NAME, which must end in `_openvino_model`, so
    the staged copy keeps that suffix.

    TensorRT is exported fp32 so its parity is comparable with the others; fp16
    would be faster but would be measuring a different model.
    """
    kw = {"onnx": {"opset": 17, "simplify": True},
          "engine": {"half": False},
          "openvino": {}}[fmt]
    src = str(YOLO(pt).export(format=fmt, imgsz=size, **kw)).rstrip("\\/")
    if fmt == "openvino":
        dst = os.path.join(STAGE, f"ov_{size}", "best_openvino_model")
    else:
        dst = os.path.join(STAGE, f"{fmt}_{size}{os.path.splitext(src)[1]}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sizes", default="640,416")
    ap.add_argument("--skip", default="", help="label dipisah koma")
    a = ap.parse_args()
    os.chdir(ROOT)
    sizes = [int(x) for x in a.sizes.split(",")]
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}

    from ultralytics import YOLO
    import torch

    pt = os.path.join(ROOT, "runs", "person", a.run, "weights", "best.pt")
    if not os.path.exists(pt):
        sys.exit(f"bobot tidak ada: {pt}")
    paths = sample(a.n)
    todo = [b for b in BACKENDS
            if b[0] not in skip and (torch.cuda.is_available() or b[3] != "GPU")]
    res = {"run": a.run, "conf": CONF, "n_images": len(paths), "sizes": sizes,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "hw_of": {b[0]: b[3] for b in todo},
           "backends": {}, "parity": {}}
    os.makedirs(STAGE, exist_ok=True)

    # one export per (format, size), shared by every backend that uses it
    print("=== export ===")
    by_fmt = {}
    model_of = {}
    for label, fmt, dev, hw in todo:
        for z in sizes:
            if fmt is None:
                model_of[(label, z)] = pt
                continue
            if (fmt, z) not in by_fmt:
                try:
                    by_fmt[(fmt, z)] = export_one(YOLO, pt, fmt, z)
                    print(f"  {fmt:<9}{z}px -> "
                          f"{os.path.basename(by_fmt[(fmt, z)])}")
                except Exception as e:
                    by_fmt[(fmt, z)] = None
                    print(f"  {fmt:<9}{z}px GAGAL {type(e).__name__}: {str(e)[:80]}")
            model_of[(label, z)] = by_fmt[(fmt, z)]
    res["artifacts"] = {f"{f}_{z}": p for (f, z), p in by_fmt.items() if p}

    print(f"\n=== parity vs PyTorch CPU @640 ({len(paths)} gambar) ===")
    ref = detect(YOLO(pt), paths, "cpu", 640, 8)
    for label, fmt, dev, hw in todo:
        if label == "pytorch_cpu":
            continue
        mp = model_of.get((label, 640))
        if not mp:
            continue
        try:
            m = YOLO(mp) if fmt is None else YOLO(mp, task="detect")
            p = parity(ref, detect(m, paths, dev, 640, 8 if fmt is None else 1))
            res["parity"][label] = p
            print(f"  {label:<14}[{hw}] {p['boxes']:>4} kotak  cocok "
                  f"{p['match_pct']:>5.1f}%  dconf {p['max_conf_delta']:.5f}  "
                  f"identik {p['count_equal_images']}/{len(paths)}")
            del m
        except Exception as e:
            res["parity"][label] = {"error": f"{type(e).__name__}: {e}"[:200]}
            print(f"  {label:<14}[{hw}] GAGAL {type(e).__name__}: {str(e)[:70]}")
        gc.collect()

    print(f"\n=== kecepatan ({len(paths)} gambar, 1 frame/panggilan) ===")
    print(f"  {'backend':<14}{'hw':>5}" + "".join(f"{f'{z}px':>12}" for z in sizes))
    for label, fmt, dev, hw in todo:
        row = {}
        for z in sizes:
            mp = model_of.get((label, z))
            if not mp:
                continue
            # per-size try/except: one resolution failing must not discard a
            # measurement that already succeeded at another
            try:
                m = YOLO(mp) if fmt is None else YOLO(mp, task="detect")
                row[z] = bench(m, paths, dev, z)
                del m
                gc.collect()
            except Exception as e:
                row[f"error_{z}"] = f"{type(e).__name__}: {e}"[:200]
        res["backends"][label] = row
        cells = "".join(f"{row[z]:>9.1f} ms" if z in row else f"{'-':>12}"
                        for z in sizes)
        print(f"  {label:<14}{hw:>5}{cells}")

    json.dump(res, open(f"{OUT}/backends_{a.run}.json", "w"), indent=1)
    print(f"\ndisimpan -> reports/eval/backends_{a.run}.json")


if __name__ == "__main__":
    main()
