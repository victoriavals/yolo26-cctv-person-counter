# Project memory: yolo26-cctv-person-counter

Entry point for AI agents. Read this file first, then whichever linked file fits
the task. Everything here was checked against the code and the files on disk on
**2026-09-23**, at commit `b3cef02` plus the uncommitted changes listed under
[Repository state](#repository-state). Anything that could not be checked says so.

| File | Read it when |
|---|---|
| [architecture.md](architecture.md) | adding a pipeline step, or tracing which script feeds which |
| [database.md](database.md) | reading or writing any dataset, label, metric or cache file |
| [conventions.md](conventions.md) | writing or editing code in `tools/` |
| [decisions.md](decisions.md) | before changing a threshold, split, augmentation or backend choice |
| [commands.md](commands.md) | running anything |
| [features.md](features.md) | checking what exists, what is half-built and what is missing |

## How this folder relates to the other docs

| File | Audience | In git | Notes |
|---|---|---|---|
| `README.md` | developers, public | yes | long-form rationale and every results table |
| `PANDUAN.md` | the store operator, in Indonesian | yes | step-by-step guide for Stage 6 |
| `CLAUDE.md` | Claude Code, loaded automatically | **no** (in `.gitignore`) | local working notes, partly out of date |
| `.agents/` | any AI agent | yes | this set: a checked summary, plus the gaps |

When they disagree, trust the code first, then this folder, then `README.md`,
then `CLAUDE.md`. The known disagreements are listed under
[Known stale claims](#known-stale-claims).

## Project identity

| | |
|---|---|
| Name | `yolo26-cctv-person-counter` (GitHub `victoriavals/yolo26-cctv-person-counter`; local folder `person-counter`) |
| Purpose | Count people in an **indoor store** from a **fixed ceiling-mounted CCTV** camera |
| Type | Offline ML pipeline: dataset cleaning, training, evaluation, export and counting, as standalone Python CLI scripts. There is no service, API, UI app or database |
| Target users | A store operator who records their own camera and labels frames (`PANDUAN.md` is written for them). Which store, and what hardware it has: Unknown - information not available in repository |
| Framework | Ultralytics 8.4.60 (YOLO26s, one class `person`) on PyTorch 2.14.0+cu126 |
| Language | Python 3.11.9, system-wide install, no virtualenv |
| Database | None. All data is files; see [database.md](database.md) |
| Infrastructure | One Windows 11 workstation: Intel i5-13400F, NVIDIA RTX 4060 Ti 8 GB, 34 GB RAM |
| Hosting / deployment | Unknown - information not available in repository. No deployment config, service or live-stream code exists |
| Package manager | pip, but **there is no dependency manifest** (no `requirements.txt` or `pyproject.toml`). The versions in [commands.md](commands.md) were read from the live environment |
| Dev tools | git (remote on GitHub); inference runtimes TensorRT 11.3.0.99, OpenVINO 2026.4.0, ONNX Runtime 1.21.1 (CPU only); ffmpeg used by hand to re-encode videos, not called by any tool |

## Status by stage

| Stage | Work | Status |
|---|---|---|
| 1 | Audit and clean the Roboflow export | Done |
| 2 | Train `default`, `tuned`, `indoor` | Done |
| 3 | Evaluate on the held-out test split | Done |
| 4 | Pick the model: `default` at `conf=0.25`, 640px | Done |
| 5 | Export, backend benchmark, counting demo | Done |
| 6 | Own-store footage: sample, label, merge, fine-tune, evaluate | **Blocked on footage.** Sampling, merge and training are built. **Evaluation on store frames is not** |

## Critical knowledge

These are the facts most likely to cost an agent hours if missed.

1. **The model does not work on real ceiling CCTV yet.** On a 1920x1080 warehouse
   clip it reported 2.35 people against 5-7 visible (60-80% missed) and gave 40
   track IDs to about 6 people. Raising `imgsz` from 640 to 1600 changed nothing
   (15 -> 15 detections). The cause is **seated and table-occluded people**, a shape
   missing from the training data. A test-split mAP50 of 0.891 did not predict this.
2. **The model to use** is `runs/person/default/weights/best.pt` at `conf=0.25`,
   `imgsz=640`. It is not in git; it is meant to be published as a GitHub Release asset.
3. **The exports next to that model are 416px, not 640px.** `weights/best.engine`,
   `weights/best.onnx` and `weights/best_openvino_model/` were left there by
   `tools/benchmark_backends.py`, which exports every size beside the weights and
   the last size, 416, wins. Loading one and asking for 640 silently runs at 416
   (measured: 13.2% fewer boxes on a crowded clip). The correct 640 files are
   `reports/eval/backends/engine_640.engine`, `onnx_640.onnx` and
   `ov_640/best_openvino_model/`, plus copies in `runs/person/default/trt640/` and
   `ov640/`. Check a graph's baked size before using it (see [commands.md](commands.md#debugging)).
4. **Set `os.environ["YOLO_AUTOINSTALL"] = "False"` before importing ultralytics.**
   Otherwise loading an `.onnx` model with CUDA present triggers
   `pip install onnxruntime-gpu`, which fails halfway on this Python and breaks
   ONNX Runtime.
5. **Pin OpenVINO devices as `intel:cpu` or `intel:gpu`.** Plain `device="cpu"`
   can be sent to OpenVINO's `AUTO`, which may pick the GPU.
6. **ONNX Runtime has no GPU provider here** (`CPUExecutionProvider` only). Do not
   install `onnxruntime-gpu`; see item 4.
7. **Stage 6 evaluation does not exist.** `evaluate.py` and `make_eval_figures.py`
   only know `baseline, default, tuned, indoor` and only read the base COCO ground
   truth. `PANDUAN.md` step 7 and the file `errors_store.png` describe something no
   tool produces. See [features.md](features.md#store-frame-evaluation).
8. **Backend parity is not universal.** On the 40 test images all six backends agree
   exactly (155/155 boxes). On a 240-frame crowded video segment, ONNX, OpenVINO
   CPU and OpenVINO GPU agree with each other on 240/240 frames but with PyTorch on
   only 151-152/240, and find about 2.8% more boxes, mostly small, distant people.
   Also note that the `match_pct` parity metric only measures how many reference
   boxes were found, so it cannot detect a backend that **adds** boxes.
9. **YOLO26s is end-to-end and NMS-free** (`end2end=True`, `reg_max=1`). The only
   threshold to tune is `conf`; there is no NMS `iou`.
10. **Never split frames from one camera at random.** Neighbouring frames are near
    duplicates and leak across splits. Split by scene (base data) or by contiguous
    time block (store data). This exact leak made the original Roboflow split
    useless.
11. **A relative `project=` in ultralytics lands in another project.** It resolves
    against the global `runs_dir` in `%APPDATA%\Ultralytics\settings.json`, which
    points at an unrelated repo. Always pass an absolute path.
12. **OpenCV cannot open Git Bash paths.** `cv2.VideoCapture("/d/ML/...")` fails
    without an error; use `D:/ML/...`.
13. **Draft labels are model output, not ground truth.** Never train or evaluate on
    `data/store/*/labels/` before a human has corrected them.

## Known stale claims

| Claim | Where | What is actually true (checked 2026-09-23) |
|---|---|---|
| "Not a git repository" | `CLAUDE.md` | Git repo on branch `main`, one commit `b3cef02` (2026-09-22), pushed to `origin` |
| `images/` holds "4886 unique photos" | `CLAUDE.md` layout | `data/cctv-person/images/` holds **4746**: the 4886 distinct photos in the export minus 140 dropped `youtube-*` |
| "All six backends are exactly faithful ... identical counts on 40/40 images" | `README.md`, `CLAUDE.md` | True on those 40 test images, false on dense video; see critical item 8 |
| Stage 6 "is tooled" | `CLAUDE.md` | Sampling, merge and training are; store evaluation, `errors_store.png` and monitoring the `store` run are not |
| Monitor `train.py store` with `status.py --watch` | `PANDUAN.md` step 6 | `status.py` lists only `default, tuned, indoor` |
| `evaluate.py --all` gives the store-frame number | `PANDUAN.md` step 7 | No `store` target, and no way to read `data/merged/` |
| "Copy the COMMON block, change `data=`" | `merge_store_data.py` final output | Obsolete: `python tools/train.py store` already exists |
| Store validation share defaults to 15% | `merge_store_data.py --store-val` | It is 20%: `round(10 * 0.15)` is `round(1.5)`, which Python rounds to 2 blocks. Test is 20%, train 60% |
| "No images are committed here" | `README.md` Licence | Derived images **are** committed: `reports/figures/sheet_cctv_*.png`, `fig_leakage.png`, `rhc_preview.png`, the `grid_*`/`errors_*` galleries and the tracking previews |
| "Reusing the 640 graph at 416 silently keeps running at 640" | `README.md`, `CLAUDE.md` | The reverse is also true and was hit: a 416 graph asked for 640 runs at 416, silently |

## Repository state

- Branch `main`, one commit, local equal to `origin/main`. Whether the GitHub repo is
  public: Unknown - information not available in repository.
- Uncommitted on 2026-09-23: `tools/compare_backends_video.py`,
  `reports/tracking/backends.html`, `reports/eval/backend_compare/`, and this
  `.agents/` folder.
- Large files that are present but ignored: `archieved_dataset/` (598 MB source zip),
  `data/raw/`, `data/cctv-person/images/`, all weights and exports, all `.mp4`,
  `data/video/people-walking.mp4`, and two stray downloads in the root:
  `vt5c8h6kmh-1.zip` (the third-party RHC dataset) and `yolo26n.pt`.

## Important files

| File | Role |
|---|---|
| `tools/families.py` | The only definition of scene-family regexes and `DROP_KINDS` |
| `tools/train.py` | `COMMON` config and the `RUNS` table: every experiment, including `store` |
| `tools/evaluate.py` | Test-split mAP (pycocotools) and counting MAE, by domain and density |
| `tools/count_people.py` | Detect, track and count over a video |
| `tools/prepare_labeling.py`, `tools/merge_store_data.py` | Stage 6 data tools |
| `data/cctv-person/annotations/{train,val,test}.json` | COCO ground truth, committed |
| `data/cctv-person/manifest.csv` | One row per photo: scene, split, domain kind, density |
| `data/cctv-person/yolo/data.yaml` | Training config, with an **absolute** `path:` |
| `reports/dup_audit.json` | Duplicate clusters that `clean_cctv.py` depends on |
| `reports/eval/*.json` | Every measured number behind the decisions |

## Do not change casually

- `data/cctv-person/annotations/*.json`, `manifest.csv`, and `yolo/labels/`: ground
  truth. Regenerate them through the pipeline, never by hand.
- `reports/dup_audit.json`: `clean_cctv.py` reads its cluster IDs. Re-running the
  audit changes which copy represents each photo.
- `tools/families.py`: changing a pattern changes the dataset, the split and every
  domain breakdown.
- `COMMON` in `tools/train.py`: changing it makes new runs incomparable with the
  existing four.
- `runs/person/default/weights/best.pt`: the chosen model, not in git.
- `.gitignore`: it deliberately keeps the COCO JSON, manifest and YAML while
  ignoring images and weights.

## AI agent rules

1. Read `.agents/memory.md` before making changes.
2. Understand the pipeline in [architecture.md](architecture.md) before adding a feature.
3. Follow the existing patterns in [conventions.md](conventions.md).
4. Do not add a dependency without a reason; there is no manifest to record it in,
   so write it into [commands.md](commands.md) as well.
5. Do not refactor at scale without an impact analysis. Many scripts hard-code run
   names and paths (see [features.md](features.md)).
6. Do not delete a feature, script, run or report without the user's confirmation.
7. Keep backward compatibility of file formats: existing JSON and CSV in `reports/`
   are read by other scripts.
8. Mind security and privacy: CCTV frames show identifiable people. Do not upload
   footage to third-party services, commit new images, or bind servers to network
   interfaces without the user's decision.
9. Ask when information is not available, rather than assuming.
10. Update this folder when something significant changes.

Rules specific to this repo:

11. **Measure before claiming.** Every decision here rests on a number in `reports/`.
    A new claim needs a new measurement, and a failed hypothesis gets recorded in
    [decisions.md](decisions.md).
12. **Do not retry the two failed hypotheses** (lower augmentation; indoor-only
    training) without new evidence.
13. **Do not tune the tracker** until detection is fixed; broken detection is what
    fragments the tracks.
14. **Re-tune `conf` whenever `imgsz` changes.** The best threshold moves with size.
15. **Stage every export in its own directory**, and verify its baked `imgsz`.
16. Report counting MAE alongside mAP. The product counts people.

## Development guidelines

- Run scripts from the repo root. Three older scripts (`audit_duplicates.py`,
  `clean_cctv.py`, `make_figures.py`) use paths relative to the working directory.
- Write new scripts with an absolute `ROOT`, an argparse `main()`, and a module
  docstring that says why the script exists and how to call it.
- Put the reasoning for a number in a comment next to it, including what was
  measured. The existing code does this throughout.
- Keep stdout ASCII; the console is cp1252.
- Use `train.py smoke` before any long training run.
