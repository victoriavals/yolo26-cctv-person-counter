"""Chart the backend comparison from reports/eval/backends_<run>.json.

Three panels, because speed alone would be the wrong story:

  left   latency per frame on a LOG axis -- the spread runs 11 ms to 700 ms,
         and a linear axis would flatten everything below ONNX into noise
  middle throughput, with a 15 fps reference line so the number connects to
         whether it keeps up with a live CCTV camera
  right  parity against the PyTorch CPU reference -- a fast backend that drops
         detections is not a usable backend

CPU and GPU backends are separated on the x-axis: they answer different
questions ("what if the store box has no GPU" vs "what if it does").

    python tools/make_backend_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL = os.path.join(ROOT, "reports", "eval")
FIG = os.path.join(EVAL, "figures")

LABEL = {"pytorch_gpu": "PyTorch", "pytorch_cpu": "PyTorch",
         "onnx_cpu": "ONNX", "openvino_cpu": "OpenVINO",
         "openvino_gpu": "OpenVINO", "tensorrt_gpu": "TensorRT"}
COLOR = {"pytorch_gpu": "#2a78d6", "pytorch_cpu": "#2a78d6",
         "onnx_cpu": "#d2372f", "openvino_cpu": "#eb6834",
         "openvino_gpu": "#eb6834", "tensorrt_gpu": "#12805a"}
# CPU group first: that is the constrained case the deployment decision hinges on
ORDER = ["pytorch_cpu", "onnx_cpu", "openvino_cpu",
         "pytorch_gpu", "openvino_gpu", "tensorrt_gpu"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
})


def val(be, b, z):
    v = be[b].get(str(z), be[b].get(z))
    return v if isinstance(v, (int, float)) else None


def main():
    p = f"{EVAL}/backends_default.json"
    if not os.path.exists(p):
        sys.exit(f"tidak ada {p} -- jalankan tools/benchmark_backends.py dulu")
    d = json.load(open(p))
    os.makedirs(FIG, exist_ok=True)

    be, par, hw = d["backends"], d.get("parity", {}), d.get("hw_of", {})
    sizes = d["sizes"]
    have = [b for b in ORDER if b in be and any(val(be, b, z) for z in sizes)]
    if not have:
        sys.exit("tidak ada backend yang berhasil diukur")

    # a visual gap between the CPU block and the GPU block
    pos, gap = [], 0
    for i, b in enumerate(have):
        if i and hw.get(b) != hw.get(have[i - 1]):
            gap += 0.9
        pos.append(i + gap)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    w = 0.8 / len(sizes)

    def group_ticks(axis):
        axis.set_xticks([x + 0.4 - w / 2 for x in pos])
        axis.set_xticklabels([f"{LABEL[b]}\n{hw.get(b, '')}" for b in have],
                             fontsize=8.5)

    for k, z in enumerate(sizes):
        vals = [val(be, b, z) or 0 for b in have]
        bars = ax[0].bar([x + k * w for x in pos], vals, w,
                         color=[COLOR[b] for b in have],
                         alpha=1.0 if k == 0 else 0.5, label=f"imgsz {z}")
        for bar, v in zip(bars, vals):
            if v:
                ax[0].text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                           f"{v:.0f}", ha="center", va="bottom",
                           fontsize=8.5, fontweight="bold")
    ax[0].set_yscale("log")
    ax[0].set_ylim(6, 1600)
    ax[0].set_title("Latensi per frame - skala log (makin rendah makin baik)")
    ax[0].set_ylabel("ms / frame")
    group_ticks(ax[0])
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")

    for k, z in enumerate(sizes):
        vals = []
        for b in have:
            v = val(be, b, z)
            vals.append(1000 / v if v else 0)
        bars = ax[1].bar([x + k * w for x in pos], vals, w,
                         color=[COLOR[b] for b in have],
                         alpha=1.0 if k == 0 else 0.5, label=f"imgsz {z}")
        for bar, v in zip(bars, vals):
            if v:
                ax[1].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                           f"{v:.1f}", ha="center", va="bottom",
                           fontsize=8.5, fontweight="bold")
    ax[1].axhline(15, color="#444", ls="--", lw=1.2)
    ax[1].text(max(pos) + 0.7, 17, "15 fps = kamera CCTV live",
               ha="right", fontsize=8.5, color="#444")
    ax[1].set_title("Throughput")
    ax[1].set_ylabel("FPS")
    ax[1].set_ylim(0, 105)
    group_ticks(ax[1])

    pk = [b for b in have if b in par and "match_pct" in par[b]]
    vals = [par[b]["match_pct"] for b in pk]
    bars = ax[2].bar(range(len(pk)), vals, 0.6, color=[COLOR[b] for b in pk])
    for bar, b, v in zip(bars, pk, vals):
        ax[2].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                   f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax[2].text(bar.get_x() + bar.get_width() / 2, 5,
                   f"dconf\n{par[b]['max_conf_delta']:.5f}", ha="center",
                   fontsize=7.2, color="white", fontweight="bold")
    ax[2].axhline(100, color="#444", ls="--", lw=1.2)
    ax[2].set_ylim(0, 115)
    ax[2].set_title("Kesetiaan vs PyTorch CPU (IoU>=0.95)")
    ax[2].set_ylabel("% kotak identik")
    ax[2].set_xticks(range(len(pk)))
    ax[2].set_xticklabels([f"{LABEL[b]}\n{hw.get(b, '')}" for b in pk], fontsize=8.5)

    gpu = d.get("gpu") or "GPU"
    fig.suptitle(f"Perbandingan backend inferensi - YOLO26s pada i5-13400F / {gpu}",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/backends.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("backends.png")

    print(f"\n{'backend':<15}{'hw':>4}" + "".join(f"{f'{z}px':>11}{'FPS':>7}"
                                                  for z in sizes) + f"{'parity':>9}")
    for b in have:
        cells = ""
        for z in sizes:
            v = val(be, b, z)
            cells += f"{v:>8.1f} ms{1000 / v:>7.1f}" if v else f"{'-':>11}{'-':>7}"
        pc = par.get(b, {}).get("match_pct")
        print(f"{b:<15}{hw.get(b, ''):>4}{cells}"
              + (f"{pc:>8.1f}%" if pc is not None else f"{'(ref)':>9}"))


if __name__ == "__main__":
    main()
