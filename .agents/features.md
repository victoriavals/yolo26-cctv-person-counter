# Feature Documentation

Status values: **Completed**, **In Progress**, **Planned**, **Unknown**. "Completed"
means the code exists and has produced the outputs described. It does not mean the
result is good enough for a store; see [memory.md](memory.md#critical-knowledge).

| Feature | Status |
|---|---|
| [Duplicate audit](#duplicate-audit) | Completed |
| [Dataset cleaning and re-split](#dataset-cleaning-and-re-split) | Completed |
| [Dataset figures and report](#dataset-figures-and-report) | Completed |
| [Training experiments](#training-experiments) | Completed |
| [Training progress monitor](#training-progress-monitor) | Completed, with a gap |
| [Test-split evaluation](#test-split-evaluation) | Completed |
| [Evaluation figures](#evaluation-figures) | Completed |
| [ONNX export with parity check](#onnx-export-with-parity-check) | Completed |
| [Multi-backend benchmark](#multi-backend-benchmark) | Completed, with a side effect |
| [Backend chart](#backend-chart) | Completed |
| [People counting with tracking](#people-counting-with-tracking) | Completed as a demonstration |
| [Frame sampling for labelling](#frame-sampling-for-labelling) | Completed |
| [Store data merge](#store-data-merge) | Completed, with caveats |
| [Store fine-tune run](#store-fine-tune-run) | Planned (blocked on footage) |
| [Store-frame evaluation](#store-frame-evaluation) | **Planned, not implemented** |
| [Tracker quality measurement](#tracker-quality-measurement) | Planned |
| [Backend comparison videos](#backend-comparison-videos) | Completed, not committed |
| [Report server](#report-server) | Completed |
| [Report pages](#report-pages) | Completed |
| [Live camera deployment](#live-camera-deployment) | Unknown |

---

## Duplicate audit

Purpose: Find which of the 10062 exported files are copies of the same photo, by
pixel content.

Status: Completed

Location: `tools/audit_duplicates.py`

Important Files: `reports/dup_audit.json` (committed), `data/raw/cctv-person/`

Dependencies: numpy, Pillow; the raw export extracted into `data/raw/cctv-person/`

Notes: Found 4886 distinct photos, and that 1651 photos leak across the exported
splits. Uses paths relative to the working directory. Takes about 3 minutes.

## Dataset cleaning and re-split

Purpose: Build the clean dataset: one copy per photo, one class, a scene-grouped
stratified 80/10/10 split, and a YOLO mirror.

Status: Completed

Location: `tools/clean_cctv.py`, `tools/families.py`

Important Files: `data/cctv-person/annotations/*.json`, `manifest.csv`, `yolo/data.yaml`

Dependencies: `reports/dup_audit.json`; numpy, Pillow

Notes: Keeps the copy with the least rotation fill. Drops `youtube-*`, shoe classes 4
and 5, and 2 degenerate boxes. Adds `__2` to case-colliding names. **Never deletes old
output**, so clear it before re-running (see [database.md](database.md#migration-strategy)).
Uses cwd-relative paths.

## Dataset figures and report

Purpose: Show the export's defects and the cleaned dataset's make-up.

Status: Completed

Location: `tools/make_figures.py`

Important Files: `reports/figures/fig_leakage.png`, `fig_domain.png`,
`sheet_cctv_*.png`, `reports/stats.json`, `reports/dataset-report.html`

Dependencies: the cleaned dataset and the raw export; `families.py`

Notes: The HTML page was written by hand from these outputs; no tool generates it.
The contact sheets contain dataset photos and are committed despite the unverified
licence.

## Training experiments

Purpose: Train YOLO26s variants from one declarative table.

Status: Completed. Four runs exist:

| Run | Epochs run | Time | Overrides |
|---|---|---|---|
| `smoke` | 2 | about 1 min | `epochs=2`, `fraction=0.08`, no cache |
| `default` | 131 (early stop, `patience=30`) | about 150 min | none |
| `tuned` | 150 | about 170 min | lighter HSV and scale augmentation, `close_mosaic=20` |
| `indoor` | 80 | about 40 min | indoor-only train set plus the `tuned` augmentation |

Location: `tools/train.py`

Important Files: `runs/person/<run>/{args.yaml, results.csv, run_meta.json, weights/best.pt}`

Dependencies: `data/cctv-person/yolo/data.yaml`, `models/yolo26s.pt`; ultralytics,
torch with CUDA

Notes: `all` means `default, tuned, indoor`; `smoke` and `store` run only when named.
Running `indoor` rewrites `data_indoor.yaml` and `train_indoor.txt`. `default` is the
chosen model.

## Training progress monitor

Purpose: Show epoch, mAP, precision and recall for each run while it trains, wherever
ultralytics wrote it.

Status: Completed, with a gap

Location: `tools/status.py`

Important Files: `runs/person/<run>/results.csv`, `run_meta.json`

Dependencies: none beyond the standard library

Notes: `ORDER = ["default", "tuned", "indoor"]` is hard-coded, so **`smoke` and
`store` never appear**, although `PANDUAN.md` step 6 tells the operator to watch the
`store` run with it. `EPOCHS_PLANNED = 150` is also fixed; `store` plans 60. The
numbers shown are from the val split.

## Test-split evaluation

Purpose: Measure detection (mAP through pycocotools) and counting (MAE, RMSE, bias,
within +-1) on the held-out test split, broken down by domain and crowd density, and
pick the `conf` that minimises counting error.

Status: Completed

Location: `tools/evaluate.py`

Important Files: `reports/eval/{baseline,default,tuned,indoor}.json`,
`default_{320,416,512}.json`, and the matching `*_preds.npz`

Dependencies: `data/cctv-person/annotations/`, `manifest.csv`; pycocotools

Notes: Targets are hard-coded to `baseline, default, tuned, indoor`. Predicts at
`conf=0.001` so the sweep has something to sweep. Feeds images in chunks of 16 and
takes filenames from the input order, never from `r.path`.

## Evaluation figures

Purpose: Curves, conf sweep, count scatter, per-domain breakdown, reliability, and
best/worst and error galleries.

Status: Completed

Location: `tools/make_eval_figures.py`

Important Files: `reports/eval/figures/*.png`, `reports/training-report.html`

Dependencies: `reports/eval/<run>.json` and `_preds.npz`; imports matching helpers from
`evaluate.py`

Notes: `ORDER` is hard-coded to the same four runs, and ground truth is always the base
`test.json`. It does not run inference.

## ONNX export with parity check

Purpose: Export to ONNX, prove the export gives the same detections, and time it.

Status: Completed

Location: `tools/export.py`

Important Files: `reports/eval/export_default.json`, `runs/person/<run>/weights/best.onnx`

Dependencies: onnx, onnxruntime; `YOLO_AUTOINSTALL=False`

Notes: Exports at opset 12, where the benchmark uses opset 17. Recorded parity is 145/145
boxes on 30 images. The ONNX file it writes next to the weights has since been
overwritten by the benchmark's 416 export.

## Multi-backend benchmark

Purpose: Measure parity and speed for six backends (PyTorch, ONNX, OpenVINO on CPU;
PyTorch, OpenVINO, TensorRT on GPU) at several sizes.

Status: Completed, with a side effect

Location: `tools/benchmark_backends.py`

Important Files: `reports/eval/backends_default.json`, `reports/eval/backends/` (the
staged exports)

Dependencies: tensorrt, openvino, onnxruntime, CUDA

Notes: **Leaves the last exported size (416) next to the weights**: `weights/best.engine`,
`best.onnx` and `best_openvino_model/` are all 416 today. Each backend and size is
wrapped in its own `try/except`. Parity uses the PyTorch CPU result as the reference.
`--run store` will work once the `store` run exists.

## Backend chart

Purpose: Plot latency (log axis), throughput against a 15 fps line, and parity.

Status: Completed

Location: `tools/make_backend_figure.py`

Important Files: `reports/eval/figures/backends.png`

Dependencies: `reports/eval/backends_default.json`

Notes: The input file name is hard-coded to `backends_default.json`, so a
`benchmark_backends.py --run store` result would not be charted.

## People counting with tracking

Purpose: Detect, track (ByteTrack or BoT-SORT) and count over a video: occupancy per
frame, and crossings of a line in each direction.

Status: Completed as a demonstration. Tracking quality has never been measured,
because no data with track IDs exists.

Location: `tools/count_people.py`

Important Files: `reports/tracking/<name>_{annotated.mp4, counts.csv, summary.json}`

Dependencies: supervision, lap, ultralytics

Notes, all checked in the code:
- `imgsz=640` is hard-coded, so a 416 export silently runs at 416.
- It does **not** set `YOLO_AUTOINSTALL`, and does not pin a device, although
  `--weights` accepts any export.
- It accepts only a file path (`os.path.exists` check), not an RTSP stream.
- The output video is written at supervision's integer fps (23 instead of 23.976) in
  `mp4v`, which browsers cannot play.

Existing runs: `warehouse` (the failed reality check), and `people-walking_{pt, trt,
trt640}` on the promenade clip. `people-walking_trt` used the 416 engine by mistake and
is kept only as the record of that trap.

## Frame sampling for labelling

Purpose: Pick up to N frames from the operator's video that are worth labelling (spread
in time, visibly different, stratified by crowding) and pre-draw draft boxes.

Status: Completed. Validated end to end on the warehouse clip; those test outputs were
deleted on purpose because their labels were model drafts.

Location: `tools/prepare_labeling.py`

Important Files: `data/store/<name>/{images, labels, classes.txt, manifest.csv}`

Dependencies: OpenCV, ultralytics, `runs/person/default/weights/best.pt`

Notes: Drafts at `conf=0.15`. The crowd buckets are counted by the draft model, which
under-counts seated people, so a bucket of `6+` can be empty because the model missed
people rather than because the store was quiet.

## Store data merge

Purpose: Merge the labelled store frames into the base dataset and build a store-only
test set.

Status: Completed, with caveats

Location: `tools/merge_store_data.py`

Important Files: `data/merged/{data.yaml, data_store_test.yaml, test_store.txt, store_manifest.csv}`

Dependencies: `data/cctv-person/yolo/`, labelled frames in `data/store/<name>/`

Notes:
- Splits by 10 contiguous time blocks. With the defaults this gives test 20%, **val
  20%** (not 15%, because `round(1.5)` is 2) and train 60%.
- Test is always the first blocks, which is the start of the recording.
- Warns below 50 store test frames.
- Its closing "next steps" message is out of date: it says to edit `train.py` by
  hand, but `python tools/train.py store` already exists.
- Never deletes old output.

## Store fine-tune run

Purpose: Fine-tune the chosen model on base plus store frames, to add seated and
occluded people.

Status: Planned (blocked on footage). The configuration exists; the run has never been
executed.

Location: `tools/train.py`, `RUNS["store"]`

Important Files: `data/merged/data.yaml` (not present yet)

Dependencies: [Store data merge](#store-data-merge)

Notes: Starts from `best.pt` with `epochs=60`, `patience=15`, `lr0=0.001`,
`warmup_epochs=1.0`. `train.py` stops with a message if `data/merged/data.yaml` is missing.

## Store-frame evaluation

Purpose: Measure the `store` model on store frames only. This is the number that
predicts field behaviour, and `PANDUAN.md` defines pass thresholds for it: MAE below
0.5, within +-1 above 90%, bias between -0.3 and +0.3, recall above 0.85 on crowded
frames.

Status: **Planned, not implemented.**

Location: none yet.

Important Files: would read `data/merged/data_store_test.yaml` and `test_store.txt`.

Dependencies: [Store fine-tune run](#store-fine-tune-run)

Notes: `PANDUAN.md` step 7 tells the operator to run `evaluate.py --all` and read the
store-frame number, and to look at `reports/eval/figures/errors_store.png`. None of
that exists:
- `evaluate.py` has no `store` target.
- It reads only the base COCO JSON; store frames have YOLO labels and no COCO JSON.
- Its domain breakdown looks frames up in the base manifest, where store frames are absent.
- `make_eval_figures.py` is hard-coded to the base runs, so no tool writes `errors_store.png`.

Building it means: a `store` entry in `TARGETS`; ground truth read from YOLO labels (or
converted to COCO) for the files in `test_store.txt`; reporting the same counting
metrics; and adding `store` to `make_eval_figures.py` and `status.py`.
`ultralytics val` with `data=data/merged/data_store_test.yaml` would give mAP only, not
the counting metrics that decide deployment.

## Tracker quality measurement

Purpose: Measure ID switches and line-count accuracy.

Status: Planned. `README.md` and `CLAUDE.md` say it must be measured on the operator's
own footage; no code and no track-ID labels exist.

Location: none yet.

Important Files: none

Dependencies: labelled footage with track IDs; working detection first

Notes: Deliberately postponed. Tuning the tracker before detection works would treat a
symptom.

## Backend comparison videos

Purpose: Show the backends side by side on the same crowded clip: frame-synced (compare
boxes) and real-time race (compare speed), for CPU (PyTorch, ONNX, OpenVINO) and GPU
(PyTorch, OpenVINO, TensorRT).

Status: Completed on 2026-09-22; the script is not committed yet.

Location: `tools/compare_backends_video.py`

Important Files: `reports/eval/backend_compare/*.json` (cache),
`reports/tracking/backends_{cpu,gpu}_{sync,race}[_h264].mp4`, `reports/tracking/backends.html`

Dependencies: all runtimes; `data/video/people-walking.mp4`; the 640 exports in
`runs/person/default/trt640/` and `ov640/`

Notes: Frames 457-696 of the clip, the densest 10 s (8.44 people on average). Measured
per frame: TensorRT 8.6 ms, PyTorch GPU 21.6, OpenVINO GPU 35.2, OpenVINO CPU 49.2,
PyTorch CPU 172.1, ONNX CPU 689.7. This run is where the two parity families were found
([decisions.md](decisions.md#superseded-all-backends-are-exactly-faithful)). The
`_h264` files are ffmpeg re-encodes made by hand; the script writes `mp4v`.

## Report server

Purpose: Serve a folder over HTTP with byte-range support, so videos can be played and
seeked in a browser from a Remote-SSH session.

Status: Completed

Location: `tools/serve_reports.py`

Important Files: `reports/tracking/player.html`, `backends.html`

Dependencies: standard library only

Notes: Binds to `127.0.0.1` by default. Reach it with VSCode port forwarding.

## Report pages

Purpose: Human-readable summaries of the dataset audit, the training results and the
tracking and backend videos.

Status: Completed

Location: `reports/dataset-report.html`, `reports/training-report.html`,
`reports/tracking/player.html`, `reports/tracking/backends.html`

Important Files: the figures and videos they reference by relative path

Dependencies: none; static HTML

Notes: Written by hand, in Indonesian. `dataset-report.html` and `training-report.html`
are claude.ai artifact sources: fragments with no `<html>` wrapper that load fonts from
Google Fonts. Their published copies are linked in `CLAUDE.md`; republish to the same
URL rather than creating a new artifact. Numbers in the pages are copied in, not
generated, so they go stale when a metric changes.

## Live camera deployment

Purpose: Run the counter continuously on the store's camera.

Status: Unknown - information not available in repository. No document commits to it,
and no code for it exists: no stream input, service, persistence of counts, or
dashboard.

Location: none

Important Files: none

Dependencies: a model that works on the store camera ([Store-frame evaluation](#store-frame-evaluation))

Notes: `PANDUAN.md` step 8 ("Pasang") runs `count_people.py` on a recorded file. The
store hardware is unknown, which is why both a GPU backend (TensorRT) and a CPU backend
(OpenVINO) were benchmarked.
