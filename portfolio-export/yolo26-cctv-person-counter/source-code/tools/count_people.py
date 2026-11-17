"""Count people in a video: detect -> track -> count.

A store wants two different numbers, and they are not the same thing:

  occupancy -- how many people are in view at this moment. Comes straight from
               detection; the model is measured for this in reports/eval/.
  footfall  -- how many distinct people crossed a line, in and out. Needs
               tracking, because the same person must not be counted twice.

Tracking is what turns the first into the second, and it is the part this
project has NO ground truth for -- the dataset carries no track IDs. So the
numbers below are a demonstration that the pipeline runs end to end, not a
measurement of how well it tracks. That measurement can only come from labelled
footage of the actual camera.

    python tools/count_people.py --source video.mp4
    python tools/count_people.py --source video.mp4 --line 0,0.55,1,0.55 --max-frames 900
    python tools/count_people.py --source video.mp4 --no-video     # stats only, faster
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import supervision as sv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "reports", "tracking")
DEFAULT_WEIGHTS = os.path.join(ROOT, "runs", "person", "default", "weights", "best.pt")
CONF = 0.25          # operating point chosen by tools/evaluate.py


def parse_line(spec, w, h):
    """'x1,y1,x2,y2' in 0..1 -> pixel Points, so the line survives a resolution change."""
    a, b, c, d = (float(v) for v in spec.split(","))
    return sv.Point(a * w, b * h), sv.Point(c * w, d * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--line", default="0,0.5,1,0.5",
                    help="x1,y1,x2,y2 ternormalisasi 0..1")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = seluruh video")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO

    src = os.path.abspath(a.source).replace("\\", "/")
    if not os.path.exists(src):
        raise SystemExit(f"video tidak ada: {src}")
    name = a.name or os.path.splitext(os.path.basename(src))[0]
    os.makedirs(OUT, exist_ok=True)

    info = sv.VideoInfo.from_video_path(src)
    total = info.total_frames if not a.max_frames else min(a.max_frames, info.total_frames)
    print(f"{os.path.basename(src)}  {info.width}x{info.height}  {info.fps} fps  "
          f"{total} frame diproses")

    start, end = parse_line(a.line, info.width, info.height)
    line = sv.LineZone(start=start, end=end)
    box_an = sv.BoxAnnotator(thickness=2)
    lab_an = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
    trace_an = sv.TraceAnnotator(trace_length=40, thickness=2)
    line_an = sv.LineZoneAnnotator(thickness=2, text_scale=0.7)

    writer = None
    if not a.no_video:
        out_mp4 = f"{OUT}/{name}_annotated.mp4"
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))

    model = YOLO(a.weights)
    rows, seen_ids, occ = [], set(), []
    t0 = time.perf_counter()

    # persist=True keeps tracker state across frames; stream=True avoids holding
    # every Result in memory for a 4000-frame video
    for i, r in enumerate(model.track(source=src, tracker=a.tracker, persist=True,
                                      conf=a.conf, imgsz=640, stream=True,
                                      verbose=False)):
        if a.max_frames and i >= a.max_frames:
            break
        det = sv.Detections.from_ultralytics(r)
        if det.tracker_id is None:
            # first frames can come back untracked; treat as detections only
            det.tracker_id = np.full(len(det), -1)
        line.trigger(det)
        live = [int(t) for t in det.tracker_id if t >= 0]
        seen_ids.update(live)
        occ.append(len(det))
        rows.append({"frame": i, "t_detik": round(i / max(info.fps, 1), 2),
                     "occupancy": len(det), "masuk": line.in_count,
                     "keluar": line.out_count, "id_kumulatif": len(seen_ids)})

        if writer is not None:
            f = r.orig_img.copy()
            labels = [f"#{int(t)}" if t >= 0 else "?" for t in det.tracker_id]
            f = trace_an.annotate(f, det)
            f = box_an.annotate(f, det)
            f = lab_an.annotate(f, det, labels)
            f = line_an.annotate(f, line)
            cv2.putText(f, f"orang: {len(det)}   unik: {len(seen_ids)}",
                        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            writer.write(f)

        if i and i % 300 == 0:
            print(f"  {i}/{total}  occupancy={len(det)}  unik={len(seen_ids)}")

    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t0
    occ = np.array(occ)

    summary = {
        "source": src, "weights": a.weights, "tracker": a.tracker, "conf": a.conf,
        "frames": len(occ), "fps_video": info.fps,
        "detik_diproses": round(len(occ) / max(info.fps, 1), 1),
        "waktu_proses_detik": round(dt, 1),
        "fps_proses": round(len(occ) / dt, 1),
        "occupancy": {"mean": float(occ.mean()), "median": float(np.median(occ)),
                      "max": int(occ.max()), "min": int(occ.min())},
        "track_unik": len(seen_ids),
        "garis_masuk": line.in_count, "garis_keluar": line.out_count,
        "garis": a.line,
    }
    json.dump(summary, open(f"{OUT}/{name}_summary.json", "w"), indent=1)
    with open(f"{OUT}/{name}_counts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    print(f"\n{'=' * 58}")
    print(f"occupancy  rata-rata {summary['occupancy']['mean']:.2f}  "
          f"median {summary['occupancy']['median']:.0f}  "
          f"maksimum {summary['occupancy']['max']}")
    print(f"track unik {summary['track_unik']}   "
          f"garis: masuk {line.in_count} / keluar {line.out_count}")
    print(f"kecepatan  {summary['fps_proses']} FPS proses "
          f"({summary['waktu_proses_detik']}s untuk {summary['detik_diproses']}s video)")
    print(f"\n-> {OUT}/{name}_summary.json, _counts.csv"
          + ("" if a.no_video else f", _annotated.mp4"))
    print("\nCATATAN: kualitas tracking TIDAK terukur di sini -- dataset ini tidak")
    print("punya track ID. Angka 'track unik' adalah demonstrasi, bukan akurasi.")


if __name__ == "__main__":
    main()
