# Architecture

## System overview

This repository is **not a service**. `tools/` holds 16 Python files: 15
command-line scripts and one shared module, `families.py`. Each script reads
files, does one job and writes files, and the scripts talk to each other only
through the filesystem.

There is no long-running process in normal use, no API, no database, no message
queue and no frontend application. The one exception is `tools/serve_reports.py`,
a small static file server used only to view reports in a browser.

Two layers do the work:

- **ML layer.** Ultralytics YOLO26s on PyTorch for training, prediction and
  tracking; pycocotools for mAP; supervision for video metadata, line counting and
  drawing.
- **Inference runtime layer.** TensorRT on the NVIDIA GPU, OpenVINO on the CPU (or an
  Intel GPU), and ONNX Runtime on the CPU. Ultralytics' `AutoBackend` chooses the
  runtime from the weights path: `.pt`, `.onnx`, `.engine`, or a directory ending in
  `_openvino_model`.

## Application flow

```
archieved_dataset/<Roboflow export>.coco.zip      10062 files, gitignored
   | unzip, by hand
   v
data/raw/cctv-person/{train,valid,test}/          original, leaky 3-way split
   |
   |-- audit_duplicates.py --------------------> reports/dup_audit.json
   |                                             (cluster ID for every raw file)
   |-- clean_cctv.py   (imports families.py, needs dup_audit.json)
   v
data/cctv-person/                                 4746 photos, 80/10/10, 1 class
   images/  annotations/{split}.json  manifest.csv
   yolo/images/{split}  yolo/labels/{split}  yolo/data.yaml
   |
   |-- make_figures.py ------------------------> reports/figures/*.png, reports/stats.json
   |                                             (reports/dataset-report.html was built from these)
   |-- train.py <run>  (writes data_indoor.yaml for the indoor run)
   v
runs/person/<run>/                                weights/best.pt, results.csv, args.yaml, run_meta.json
   |
   |-- status.py ------------------------------> progress table, stdout only
   |-- evaluate.py ----------------------------> reports/eval/<run>[_<imgsz>].json and _preds.npz
   |      '-- make_eval_figures.py -------------> reports/eval/figures/*.png
   |                                             (reports/training-report.html was built from these)
   |-- export.py ------------------------------> best.onnx and reports/eval/export_<run>.json
   |-- benchmark_backends.py ------------------> reports/eval/backends/<fmt>_<size>, backends_<run>.json
   |      '-- make_backend_figure.py -----------> reports/eval/figures/backends.png
   |-- count_people.py --source <video> -------> reports/tracking/<name>_{annotated.mp4,counts.csv,summary.json}
   '-- compare_backends_video.py --------------> reports/eval/backend_compare/*.json
                                                 reports/tracking/backends_{cpu,gpu}_{sync,race}.mp4

Stage 6, blocked on footage:

store video
   |-- prepare_labeling.py --------------------> data/store/<name>/{images, labels (drafts), classes.txt, manifest.csv}
   |-- [a human corrects the labels in CVAT, labelImg or Label Studio, then exports YOLO]
   |-- merge_store_data.py --------------------> data/merged/{images,labels}/{split}, data.yaml,
   |                                             data_store_test.yaml, test_store.txt, store_manifest.csv
   |-- train.py store -------------------------> runs/person/store/
   '-- [evaluation on store frames] -----------> NOT IMPLEMENTED
```

## Component relationships

Import graph. Everything not listed is standalone:

```
families.py  <-- clean_cctv.py, make_figures.py, train.py
evaluate.py  <-- make_eval_figures.py   (imports MATCH_IOU, iou_matrix, match)
```

File contracts, meaning who writes each artefact and who depends on it:

| Artefact | Written by | Read by |
|---|---|---|
| `reports/dup_audit.json` | `audit_duplicates.py` | `clean_cctv.py` |
| `data/cctv-person/manifest.csv` | `clean_cctv.py` | `evaluate.py` (domain kind), `train.py` (indoor list), `make_figures.py` |
| `data/cctv-person/annotations/{split}.json` | `clean_cctv.py` | `evaluate.py`, `make_eval_figures.py` |
| `data/cctv-person/yolo/data.yaml` | `clean_cctv.py` | `train.py` (`COMMON`), ultralytics |
| `runs/person/<run>/weights/best.pt` | ultralytics, via `train.py` | `evaluate.py`, `export.py`, `benchmark_backends.py`, `count_people.py`, `prepare_labeling.py`, the `store` run |
| `runs/person/<run>/results.csv` | ultralytics | `status.py`, `make_eval_figures.py` |
| `runs/person/<run>/run_meta.json` | `train.py`, after training returns | `status.py`, as the "finished" marker |
| `reports/eval/<run>.json` and `_preds.npz` | `evaluate.py` | `make_eval_figures.py` |
| `reports/eval/backends_<run>.json` | `benchmark_backends.py` | `make_backend_figure.py` (hard-coded to `default`) |
| `data/store/<name>/` | `prepare_labeling.py`, then a human | `merge_store_data.py` |
| `data/merged/data.yaml` | `merge_store_data.py` | `train.py store` |
| `data/merged/data_store_test.yaml` | `merge_store_data.py` | **nothing yet** |

## Layers that do not exist

| Layer | Status | What exists instead |
|---|---|---|
| Backend / server | Not applicable | CLI scripts. `serve_reports.py` serves static files for viewing only |
| API | Not applicable | Each script's argparse interface; see the table below |
| Controllers, services, repositories | Not applicable | One script per pipeline step; shared helpers live in `families.py` and `evaluate.py` |
| Models / entities (ORM) | Not applicable | File schemas in [database.md](database.md) |
| Middleware | Not applicable | none |
| Frontend app | Not applicable | Hand-written static HTML reports, described below |

### Command-line interfaces

| Script | Required | Main options |
|---|---|---|
| `train.py` | run names, or `all` | `--list` |
| `status.py` | none | `--watch` |
| `evaluate.py` | targets, or `--all` | `--split`, `--imgsz` |
| `export.py` | none | `--run`, `--n` |
| `benchmark_backends.py` | none | `--run`, `--n`, `--sizes`, `--skip` |
| `count_people.py` | `--source` | `--weights`, `--conf`, `--line`, `--max-frames`, `--tracker`, `--no-video`, `--name` |
| `prepare_labeling.py` | `--source` | `--n`, `--min-gap-sec`, `--weights`, `--no-predraw`, `--min-change`, `--name` |
| `merge_store_data.py` | `--store` | `--store-val`, `--store-test` |
| `compare_backends_video.py` | `--infer` and/or `--render` | `--start`, `--frames` |
| `serve_reports.py` | none | `--dir`, `--port`, `--bind` (default `127.0.0.1`) |

The others (`audit_duplicates.py`, `clean_cctv.py`, `make_figures.py`,
`make_eval_figures.py`, `make_backend_figure.py`) take no arguments.

### Report pages

| Page | Title | Built from |
|---|---|---|
| `reports/dataset-report.html` | Audit CCTV-person | `reports/figures/`, `stats.json` |
| `reports/training-report.html` | Hasil Training YOLO26s | `reports/eval/` |
| `reports/tracking/player.html` | Person Counter - TensorRT@640 | the TensorRT tracking video |
| `reports/tracking/backends.html` | Perbandingan Backend | the four comparison videos |

No tool generates these pages; they were written by hand and reference files by
relative path. Published copies of the first two exist as claude.ai artifacts
(URLs in `CLAUDE.md`). Open them through `serve_reports.py`, because browsers
cannot play the local videos from `file://` inside a Remote-SSH session.

## Inference runtime layer

| Backend | Artefact | `device=` | Constraints |
|---|---|---|---|
| PyTorch | `best.pt` | `"cpu"` or `0` | Accepts any `imgsz` and batch size |
| ONNX Runtime | `*.onnx` | `"cpu"` | CPU only here. **Batch fixed at 1.** `imgsz` baked in |
| OpenVINO | `*_openvino_model/` | `"intel:cpu"` or `"intel:gpu"` | Directory name must end in `_openvino_model`. `imgsz` baked in |
| TensorRT | `*.engine` | `0` | NVIDIA only. `imgsz` baked in. fp32 here |

A size baked into the graph wins over the `imgsz=` argument, without any warning.

## External services and integrations

| Service | Used for | Where |
|---|---|---|
| Roboflow | Source dataset (a COCO export of a forked CCTV-person project) | `archieved_dataset/`, `data/raw/` |
| Ultralytics | Model, training, prediction, tracking, export | every ML script |
| NVIDIA TensorRT | GPU inference | `benchmark_backends.py`, `compare_backends_video.py` |
| Intel OpenVINO | CPU inference | same |
| ONNX Runtime | CPU inference (measured, not recommended) | `export.py`, the benchmark scripts |
| supervision, lap | Video metadata, `LineZone`, drawing; ByteTrack and BoT-SORT association | `count_people.py` |
| pycocotools | Reference mAP | `evaluate.py` |
| GitHub | Code remote. Weights are meant to go out as Release assets; whether any release exists: Unknown - information not available in repository | `.gitignore`, `README.md` |
| claude.ai Artifacts | Published copies of the two report pages | URLs in `CLAUDE.md` |
| YouTube, via yt-dlp | One test clip, `data/video/people-walking.mp4` (listed as "Free Stock Footage For Commercial Projects") | downloaded by hand; no tool depends on it except `compare_backends_video.py` |

## Design patterns

1. **The filesystem is the interface.** Steps join on file names and fixed paths,
   never on in-memory objects, so any step can be re-run alone.
2. **One declarative experiment table.** `RUNS` in `train.py` holds every experiment
   as overrides on a shared `COMMON`, so the difference between two runs is exactly
   what is written.
3. **Compute once, render many.** `evaluate.py` saves raw predictions (`_preds.npz`),
   and the figure script draws from them without running inference again.
   `compare_backends_video.py` caches detections per backend the same way.
4. **Measure, then decide.** Each threshold, split rule and backend choice has a
   measurement in `reports/` behind it, and the reasoning sits in a comment next to
   the constant.
5. **Stage exports in their own directory**, because ultralytics writes beside the
   weights and overwrites.
6. **Split by group, never by frame**: scene units for the base data, contiguous time
   blocks for store data.
7. **Use reference implementations** (pycocotools) instead of hand-written metrics
   where one exists.

## Architecture rules

- A new pipeline step is a new script in `tools/` that reads files and writes files,
  with argparse and a docstring. Do not add a service or database without a reason
  the user has agreed to.
- Keep `families.py` the only place scene-family patterns are defined.
- Write metrics and figures to `reports/`, training output to `runs/person/`, and
  datasets to `data/`.
- Any new evaluation must report counting MAE alongside mAP, broken down by domain
  and by crowd density, like `evaluate.py` does.
- New code that loads an export must check or pin the graph's `imgsz`.
