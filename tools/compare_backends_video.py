"""Side-by-side video comparison of inference backends on one clip.

Two things are being shown, and they are not the same thing:

  sync -- every panel shows the SAME frame, so any difference in the boxes is a
          difference between backends. Speed appears only as numbers.
  race -- every panel advances at the speed that backend actually achieved, so
          a fast backend visibly runs away from a slow one. This is the one that
          answers why the backend choice matters at all.

Inference runs ONCE per backend and is cached to reports/eval/backend_compare/;
both renders read that cache, so re-rendering costs no inference.

ONNX has no GPU provider on this machine -- onnxruntime is CPU-only, see
CLAUDE.md -- so the GPU trio uses OpenVINO GPU in its place.

    python tools/compare_backends_video.py --infer --render
    python tools/compare_backends_video.py --render        # cache already built
"""
import argparse
import json
import os
import time

os.environ["YOLO_AUTOINSTALL"] = "False"   # must precede ultralytics, see CLAUDE.md

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "reports", "eval", "backend_compare")
OUT = os.path.join(ROOT, "reports", "tracking")
SRC = os.path.join(ROOT, "data", "video", "people-walking.mp4")
CONF = 0.25
IMGSZ = 640

W = os.path.join(ROOT, "runs", "person", "default").replace("\\", "/")
BACKENDS = {
    # key:          (label,      weights,                          device)
    "pytorch_cpu":  ("PyTorch",  W + "/weights/best.pt",           "cpu"),
    "onnx_cpu":     ("ONNX",     W + "/trt640/best.onnx",          "cpu"),
    "openvino_cpu": ("OpenVINO", W + "/ov640/best_openvino_model", "intel:cpu"),
    "pytorch_gpu":  ("PyTorch",  W + "/weights/best.pt",           0),
    "openvino_gpu": ("OpenVINO", W + "/ov640/best_openvino_model", "intel:gpu"),
    "tensorrt_gpu": ("TensorRT", W + "/trt640/best.engine",        0),
}
GROUPS = {
    "cpu": ("CPU  --  Intel i5-13400F", ["pytorch_cpu", "onnx_cpu", "openvino_cpu"]),
    "gpu": ("GPU  --  NVIDIA RTX 4060 Ti", ["pytorch_gpu", "openvino_gpu", "tensorrt_gpu"]),
}


def read_segment(start, n):
    cap = cv2.VideoCapture(SRC.replace("\\", "/"))
    if not cap.isOpened():
        raise SystemExit("tidak bisa membuka " + SRC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def run_backend(key, frames):
    from ultralytics import YOLO
    import torch

    label, weights, device = BACKENDS[key]
    model = YOLO(weights, task="detect")
    gpu = device == 0 or str(device).endswith("gpu")

    for f in frames[:8]:               # warmup: the first calls build the context
        model.predict(f, conf=CONF, imgsz=IMGSZ, device=device, verbose=False)

    boxes, lat, inf_ms = [], [], []
    for f in frames:
        if gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(f, conf=CONF, imgsz=IMGSZ, device=device, verbose=False)[0]
        if gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        boxes.append(r.boxes.xyxy.cpu().numpy().round(1).tolist())
        inf_ms.append(float(r.speed["inference"]))

    del model
    return {
        "key": key, "label": label, "device": str(device), "weights": weights,
        "latency": lat, "boxes": boxes,
        "ms_mean": float(np.mean(lat) * 1000),
        "ms_median": float(np.median(lat) * 1000),
        "fps": float(1.0 / np.mean(lat)),
        "ms_inference_only": float(np.mean(inf_ms)),
        "total_boxes": int(sum(len(b) for b in boxes)),
    }


# ------------------------------------------------------------------ rendering

PW, PH, BAR, HEAD = 640, 360, 104, 58       # panel video, info bar, header
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, GREY, WHITE = (120, 235, 120), (150, 150, 150), (255, 255, 255)
BOX = (235, 140, 90)


def panel(frame, res, fi, done, note):
    """One column: the frame with its boxes, plus an info bar underneath."""
    img = cv2.resize(frame, (PW, PH), interpolation=cv2.INTER_AREA)
    sx, sy = PW / frame.shape[1], PH / frame.shape[0]
    for x1, y1, x2, y2 in res["boxes"][fi]:
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)),
                      (int(x2 * sx), int(y2 * sy)), BOX, 2)

    bar = np.full((BAR, PW, 3), 22, np.uint8)
    cv2.putText(bar, res["label"], (14, 32), FONT, 0.85, WHITE, 2)
    cv2.putText(bar, "%.0f ms   %.1f FPS" % (res["ms_mean"], res["fps"]),
                (14, 62), FONT, 0.62, GREEN, 2)
    cv2.putText(bar, "frame %d/%d   orang: %d"
                % (fi + 1, len(res["boxes"]), len(res["boxes"][fi])),
                (14, 90), FONT, 0.52, GREY, 1)
    if note:
        (tw, _), _ = cv2.getTextSize(note, FONT, 0.52, 1)
        cv2.putText(bar, note, (PW - tw - 14, 90), FONT, 0.52, GREY, 1)
    if done:
        cv2.putText(bar, "SELESAI", (PW - 150, 36), FONT, 0.72, GREEN, 2)
    col = np.vstack([img, bar])
    cv2.line(col, (PW - 1, 0), (PW - 1, PH + BAR), (55, 55, 55), 1)
    return col


def header(width, title, mode_note, t_note):
    h = np.full((HEAD, width, 3), 15, np.uint8)
    cv2.putText(h, title, (14, 38), FONT, 0.8, WHITE, 2)
    (tw, _), _ = cv2.getTextSize(mode_note, FONT, 0.6, 1)
    cv2.putText(h, mode_note, (width // 2 - tw // 2, 37), FONT, 0.6, GREY, 1)
    (tw, _), _ = cv2.getTextSize(t_note, FONT, 0.68, 2)
    cv2.putText(h, t_note, (width - tw - 14, 38), FONT, 0.68, GREEN, 2)
    return h


def render(group, mode, frames, results, fps_out=24.0):
    title, keys = GROUPS[group]
    res = [results[k] for k in keys]
    n = len(frames)
    width = PW * len(res)

    if mode == "sync":
        steps = n
        mode_note = "frame sinkron -- semua panel pada frame yang sama"
    else:
        cum = [np.cumsum(r["latency"]) for r in res]
        steps = int(np.ceil(max(c[-1] for c in cum) * fps_out))
        mode_note = "waktu nyata -- tiap panel maju sesuai kecepatan aslinya"

    path = os.path.join(OUT, "backends_%s_%s.mp4" % (group, mode))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                         (width, HEAD + PH + BAR))
    for s in range(steps):
        t = s / fps_out
        if mode == "sync":
            idx, done = [s] * len(res), [False] * len(res)
        else:
            idx, done = [], []
            for c in cum:
                k = int(np.searchsorted(c, t, side="right"))
                done.append(k >= n)
                idx.append(min(k, n - 1))
        lead = int(np.argmax(idx))
        cols = [panel(frames[idx[j]], r, idx[j], done[j],
                      "memimpin" if (mode == "race" and j == lead) else "")
                for j, r in enumerate(res)]
        vw.write(np.vstack([header(width, title, mode_note, "t = %5.2f s" % t),
                            np.hstack(cols)]))
    vw.release()
    return path, steps / fps_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=457)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    frames, fps = read_segment(a.start, a.frames)
    print("segmen: frame %d..%d  (%d frame, %.2f detik @ %.3f fps)"
          % (a.start, a.start + len(frames) - 1, len(frames), len(frames) / fps, fps))

    results = {}
    for key in BACKENDS:
        cf = os.path.join(CACHE, key + ".json")
        if a.infer or not os.path.exists(cf):
            print("  menjalankan %-14s ..." % key, end="", flush=True)
            r = run_backend(key, frames)
            json.dump(r, open(cf, "w"))
            print("  %7.1f ms/frame  %6.1f FPS  %5d box"
                  % (r["ms_mean"], r["fps"], r["total_boxes"]))
        results[key] = json.load(open(cf))

    print("\n%-14s %-10s %10s %8s %9s %7s" %
          ("backend", "device", "ms/frame", "FPS", "inferensi", "box"))
    print("-" * 62)
    for k, r in results.items():
        print("%-14s %-10s %10.1f %8.1f %9.1f %7d"
              % (k, r["device"], r["ms_mean"], r["fps"],
                 r["ms_inference_only"], r["total_boxes"]))

    if a.render:
        print()
        for group in GROUPS:
            for mode in ("sync", "race"):
                p, dur = render(group, mode, frames, results)
                print("  %-42s %6.1f detik" % (os.path.relpath(p, ROOT), dur))


if __name__ == "__main__":
    main()
