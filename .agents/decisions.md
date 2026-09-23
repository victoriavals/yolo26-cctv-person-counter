# Technical Decision Records

How to read the dates: an exact date means the decision was made that day in a
recorded session. **"On or before"** means the date is taken from the modification
time of the file that implements or records the decision, so the decision is at
least that old. Training-run folders were copied into this repo on 2026-09-17 at
08:19, so their timestamps do not show when training actually ran.

Records are grouped by pipeline stage. A record that turned out wrong stays here,
marked as superseded, rather than being deleted.

## Index

**Data**

- [Deduplicate by pixel content, not by filename](#deduplicate-by-pixel-content-not-by-filename)
- [Collapse categories 0-3 into `person`, drop 4 and 5](#collapse-categories-0-3-into-person-drop-4-and-5)
- [Stratified, scene-grouped 80/10/10 split](#stratified-scene-grouped-801010-split)
- [Accept that entrances (`semi`) are never tested](#accept-that-entrances-semi-are-never-tested)
- [Drop the `youtube-*` family](#drop-the-youtube--family)
- [Do not use MOT17](#do-not-use-mot17)

**Training**

- [One experiment table over a shared config](#one-experiment-table-over-a-shared-config)
- [Dataloader `workers=2` on Windows](#dataloader-workers2-on-windows)
- [Always pass an absolute `project=`](#always-pass-an-absolute-project)
- [Keep mosaic at 1.0](#keep-mosaic-at-10)
- [Keep the default augmentation (hypothesis 1 failed)](#keep-the-default-augmentation-hypothesis-1-failed)
- [Train on every domain, not indoor only (hypothesis 2 failed)](#train-on-every-domain-not-indoor-only-hypothesis-2-failed)

**Evaluation and model choice**

- [Use `default` at conf 0.25, chosen by counting error](#use-default-at-conf-025-chosen-by-counting-error)
- [Keep 640px](#keep-640px)

**Deployment**

- [TensorRT on GPU, OpenVINO on CPU, never ONNX; export fp32](#tensorrt-on-gpu-openvino-on-cpu-never-onnx-export-fp32)
- [Superseded: "all backends are exactly faithful"](#superseded-all-backends-are-exactly-faithful)
- [Own-store footage is mandatory](#own-store-footage-is-mandatory)
- [Do not tune the tracker yet](#do-not-tune-the-tracker-yet)

**Stage 6**

- [Sample frames by grayscale pixel difference, not the HSV descriptor](#sample-frames-by-grayscale-pixel-difference-not-the-hsv-descriptor)
- [Split store frames by contiguous time block](#split-store-frames-by-contiguous-time-block)
- [Fine-tune the store run from `best.pt` at a low learning rate](#fine-tune-the-store-run-from-bestpt-at-a-low-learning-rate)
- [Draft labels at conf 0.15](#draft-labels-at-conf-015)

**Repository and tooling**

- [Commit code and metrics, not images or weights; keep `CLAUDE.md` local](#commit-code-and-metrics-not-images-or-weights-keep-claudemd-local)
- [Serve reports over HTTP on localhost, with Range support](#serve-reports-over-http-on-localhost-with-range-support)
- [Backend comparison video: OpenVINO GPU in place of ONNX GPU](#backend-comparison-video-openvino-gpu-in-place-of-onnx-gpu)

---

**Stage: Data**

## Deduplicate by pixel content, not by filename

Date: on or before 2026-09-16 (`tools/audit_duplicates.py`, `reports/dup_audit.json`)

Context: The Roboflow export has 10062 files. Augmentation is baked into the pixels,
and Roboflow keeps the original filename stem, so copies of one photo share a stem.
But 1507 stems also hold **more than one distinct** photo; for example, 44 files named
`youtube-0_jpg.rf.<hash>.jpg` are different pictures.

Decision: Cluster the files by a rotation-tolerant descriptor (an HSV histogram plus a
grayscale histogram over non-fill pixels), with cosine similarity >= 0.97 counting as
the same photo, and cluster only within a stem group.

Reason: Measured separation: median similarity 0.989 within a stem against 0.514 across
stems, and a 0.30% false-positive rate on unrelated pairs at 0.97.

Impact: 4886 distinct photos were recovered. `clean_cctv.py` depends on the cluster IDs
in `dup_audit.json`.

Alternative Considered: Grouping by filename, which an earlier pass did and which lost
1711 real images. An exact hash was also rejected, because it misses rotated copies.

## Collapse categories 0-3 into `person`, drop 4 and 5

Date: on or before 2026-09-17 (recorded in `CLAUDE.md`, implemented in `clean_cctv.py`)

Context: The export's category table is left over from a dataset merge. 99.5% of the
boxes sit on category 2, which is named with a timestamp. Categories 4 and 5 are
safety shoes: 60 boxes, h/w 1.0-1.4, only in the `valid` split.

Decision: `PERSON_CATS = {0, 1, 2, 3}` become class `person`; `DROP_CATS = {4, 5}` are
removed.

Reason: Shoe boxes are not people, and their shape (h/w about 1.2, 0.3% of the frame)
would teach the wrong thing.

Impact: A single clean class. Code that looks for `name == "person"` in the raw export
finds nothing, so this mapping is the only correct entry point.

Alternative Considered: Mapping every category to `person`. An earlier pass did this
after checking only the train split, which contains nothing but category 2.

## Stratified, scene-grouped 80/10/10 split

Date: on or before 2026-09-16 (`data/cctv-person/manifest.csv`)

Context: The exported split leaked: 75% of test files and 70% of valid files share a
photo with another split. A plain greedy re-split by scene kept 80/10/10 but drifted in
make-up: 18% indoor in test against 37% overall, and a test median of 5 people per frame
against 2 in train.

Decision: `assign_splits()` assigns whole scenes inside `(domain kind x density bucket)`
strata, tracking the split deficit globally across strata.

Reason: Keeps the no-leak guarantee (a scene is never split) while holding the indoor
share and crowd density steady. Quotas per stratum alone drifted to 77/12/11.

Impact: Every split is 37.0-37.1% indoor, and the ratio is exactly 80/10/10.

Alternative Considered: A plain global greedy split (it measured the wrong domain), and
quotas per stratum (the global ratio drifted).

## Accept that entrances (`semi`) are never tested

Date: on or before 2026-09-16

Context: The `semi` family (entrances, 142 photos, 3.0%) all lands in train.

Decision: Leave it that way.

Reason: Forcing `semi` into val and test by reserving each stratum's smallest scenes was
tried and made things far worse: test swung to 60.9% outdoor and 1.3% unknown.

Impact: Entrance performance is never measured on held-out data.

Alternative Considered: Reserving small scenes per stratum (tried and rejected, above).

## Drop the `youtube-*` family

Date: on or before 2026-09-16 (`tools/families.py`, `DROP_KINDS`)

Context: 140 photos (372 files counting augmented copies) are webcams and selfies.

Decision: Kind `bukan-cctv` is dropped when `clean_cctv.py` loads the raw export.

Reason: The angle, distance and lens are wrong for a ceiling store camera.

Impact: The dataset went from 4886 to 4746 photos.

Alternative Considered: Keeping them as extra person examples. Rejected on domain grounds.

## Do not use MOT17

Date: on or before 2026-09-17 (recorded in `CLAUDE.md` and `README.md`)

Context: MOT17 has track IDs, which this dataset lacks.

Decision: It was evaluated and dropped, and the archives deleted.

Reason: 7 scenes: 6 outdoor streets and 1 indoor mall filmed from a moving head-height
camera. None matches a ceiling camera's geometry.

Impact: No tracking ground truth exists anywhere in the repo. Tracker quality can only
be measured on the operator's own labelled footage.

Alternative Considered: Using MOT17 anyway for the association half only. Rejected
because the geometry mismatch would make the numbers meaningless.

---

**Stage: Training**

## One experiment table over a shared config

Date: on or before 2026-09-17 (`tools/train.py`)

Context: Several runs have to be compared fairly.

Decision: `COMMON` holds the shared settings; each entry in `RUNS` lists only its
overrides.

Reason: Any difference between two runs is exactly the declared overrides.

Impact: Changing `COMMON` makes new runs incomparable with the existing four.

Alternative Considered: Separate command lines per run, which drift silently.

## Dataloader `workers=2` on Windows

Date: on or before 2026-09-17

Context: With the default `workers=8`, validation died with
`RuntimeError: Pin memory thread exited unexpectedly`.

Decision: Pin `workers=2`.

Reason: Training keeps a train loader and a val loader alive at once: 16 worker
processes plus pin-memory threads. Probed alone, `val()` survives 0, 2, 4 or 8 workers,
so it is the two loaders together. Images are cached in RAM, so workers only augment;
on val, 0 workers took 8 s against 17 s for 8.

Impact: Training runs without the crash.

Alternative Considered: The default of 8, which crashes. `workers=0` also worked on val.

## Always pass an absolute `project=`

Date: on or before 2026-09-17

Context: The first three runs landed in
`D:\computer-vision\density-aware-yolo26-vehicle-counting\...` instead of this repo.
Their `args.yaml` still records `project: runs/person`.

Decision: `train.py` passes an absolute path, and `status.py` checks both locations.

Reason: Ultralytics resolves a relative `project=` against its own global `runs_dir`.

Impact: The runs had to be copied back, which is why their timestamps read 2026-09-17 08:19.

Alternative Considered: Editing the global ultralytics setting. Not done, because it
belongs to another project.

## Keep mosaic at 1.0

Date: on or before 2026-09-17

Context: 50% of training frames hold two people or fewer.

Decision: `mosaic=1.0` in every run, including `tuned`.

Reason: Compositing four images per sample is the only mechanism here that raises the
number of people per sample, which is the crowded case a counter cares about.

Impact: No run has lowered it, so its effect has not been measured.

Alternative Considered: Lowering it along with the other augmentations. Rejected in advance, on the reasoning above.

## Keep the default augmentation (hypothesis 1 failed)

Date: on or before 2026-09-17 (`reports/eval/tuned.json`)

Context: The pixels are already augmented and the camera is fixed, so lighter
augmentation looked reasonable.

Decision: Keep Ultralytics defaults. The `tuned` run (`hsv_v` 0.15, `hsv_s` 0.4,
`scale` 0.3, `close_mosaic` 20) is not used.

Reason: `tuned` came out slightly worse on almost everything: mAP50 0.873 against 0.891,
MAE 0.659 against 0.640.

Impact: **Do not retry without new evidence.**

Alternative Considered: The `tuned` run itself.

## Train on every domain, not indoor only (hypothesis 2 failed)

Date: on or before 2026-09-17 (`reports/eval/indoor.json`)

Context: Outdoor data might have pulled the model away from the store domain.

Decision: Train on all domains.

Reason: `indoor` reached mAP50 0.885 on the indoor subset, below `default`'s 0.904 there,
and collapsed to 0.270 on `unknown` scenes. Outdoor data helps indoor performance.

Impact: **Do not retry without new evidence.**

Alternative Considered: The `indoor` run itself.

---

**Stage: Evaluation and model choice**

## Use `default` at conf 0.25, chosen by counting error

Date: on or before 2026-09-17 (`reports/eval/default.json`)

Context: The product counts people. The threshold that minimises counting MAE is usually
not the one that maximises F1.

Decision: `evaluate.py` sweeps `conf` from 0.05 to 0.95 and reports both optima. The
operating point is the minimum-MAE one: `default` at `conf=0.25`, 640px.

Reason: `default` wins on the test split: mAP50 0.891, MAE 0.640, +-1 88.0%, against a COCO
baseline of 0.644 and 1.600.

Impact: `CONF = 0.25` is hard-coded in `export.py`, `benchmark_backends.py`,
`count_people.py` and `compare_backends_video.py`. The best `conf` moves with `imgsz`
(0.20 at 416, 0.15 at 320), so re-tune it whenever the size changes.

Alternative Considered: The max-F1 threshold; `tuned`; `indoor`.

## Keep 640px

Date: on or before 2026-09-17 (`reports/eval/default_416.json` and the 320/512 files)

Context: Lower sizes are faster on CPU.

Decision: Stay at 640. 416 is acceptable only if more frames per second are needed, and
then with `conf=0.20`.

Reason: With OpenVINO (19.9 FPS) or TensorRT (78.3 FPS), 640 already clears a 15 fps
camera. 416 costs a little overall (mAP50 0.869) but improves indoor MAE (0.562 to
0.523). At 320, AP_small collapses to 0.173.

Impact: On GPU, 640 and 416 cost about the same.

Alternative Considered: 512, 416 and 320, all measured.

---

**Stage: Deployment**

## TensorRT on GPU, OpenVINO on CPU, never ONNX; export fp32

Date: on or before 2026-09-17 (`reports/eval/backends_default.json`)

Context: The hardware in the store is unknown.

Decision: Use TensorRT where there is an NVIDIA GPU and OpenVINO on a CPU. Never ONNX
Runtime. Export at fp32.

Reason: At 640px: TensorRT 12.8 ms against 26.8 ms for PyTorch GPU; OpenVINO CPU 50.2 ms
against 257.4 ms for PyTorch CPU and 696.2 ms for ONNX. OpenVINO on an NVIDIA GPU is
slower than PyTorch GPU. fp16 would be faster but is a different model, so parity could
not be compared.

Impact: Every export bakes in its `imgsz`, and ONNX fixes batch at 1.

Alternative Considered: ONNX at opset 17 (recovered only 7%), and OpenVINO GPU.

## Superseded: "all backends are exactly faithful"

Date: recorded on or before 2026-09-17; superseded 2026-09-22

Context: On the 40 test images, every backend matched PyTorch on 155/155 boxes with
identical counts, and this was written up as exact parity.

Decision: The claim is limited to that sample.

Reason: On 240 crowded frames of a real video, ONNX, OpenVINO CPU and OpenVINO GPU agree
with each other on 240/240 frames but with PyTorch on only 151-152/240, and find 2156
boxes against PyTorch's 2096-2097. The extra boxes are mostly small, distant people. The
cause has not been established.

Impact: The backend choice stands, since the difference is small and in the fast
backends' favour. But the parity check must also compare box **counts**, because
`match_pct` is blind to added boxes.

Alternative Considered: Not applicable.

## Own-store footage is mandatory

Date: on or before 2026-09-17 (`reports/tracking/warehouse_summary.json`)

Context: A warehouse packing-room clip, 900 frames, ceiling camera at about 35 degrees.

Decision: Stage 6 (the operator's own labelled footage) is a requirement, not an option.

Reason: Occupancy averaged 2.35 against 5-7 visible, and the tracker produced 40 IDs for
about 6 people. Sweeping `imgsz` 640 to 1600 and `conf` 0.25 to 0.08 barely moved the
detections (15 to 15 at 0.25). The model does not recognise seated, table-occluded people.

Impact: No number in `reports/eval/` is a deployment estimate.

Alternative Considered: Higher inference resolution and a lower threshold (both measured
and ruled out).

## Do not tune the tracker yet

Date: on or before 2026-09-17

Context: 40 unique IDs for about 6 people.

Decision: Leave ByteTrack at its defaults until detection works.

Reason: The fragmentation comes from detections that come and go, not from tracker settings.

Impact: The track counts in `reports/tracking/` are demonstrations, not accuracy.

Alternative Considered: Tuning the ByteTrack buffers. Rejected as treating the symptom.

---

**Stage: Stage 6**

## Sample frames by grayscale pixel difference, not the HSV descriptor

Date: on or before 2026-09-17 (`tools/prepare_labeling.py`)

Context: Near-identical frames from a fixed camera waste the labelling budget.

Decision: Compare 64x64 grayscale thumbnails; a frame is kept if its mean absolute
difference from every kept frame is above `--min-change` (default 2.0).

Reason: The HSV descriptor from `audit_duplicates.py` separates scenes, but on a fixed
camera the static background dominates it. It dropped 88 of 99 candidates, leaving 11 of
60 requested. The pixel difference kept all 99.

Impact: Stratified by the draft model's count (15/30/30/25% for 0, 1-2, 3-5, 6+). Those
counts come from a model that under-counts, so the bucket shares are approximate.

Alternative Considered: Reusing the HSV descriptor (measured and rejected).

## Split store frames by contiguous time block

Date: on or before 2026-09-17 (`tools/merge_store_data.py`)

Context: Frames seconds apart look alike. A random split would leak them, the exact
defect of the Roboflow export.

Decision: Ten contiguous blocks; the first blocks go to test and the next to val. The
test share defaults to 20%. A store-only test config (`data_store_test.yaml`) is written
alongside the merged one.

Reason: No leakage, and a test number that reflects the real camera.

Impact: With defaults the split is test 20% / **val 20%** / train 60%, because
`round(1.5)` is 2. Test is always the **start** of the recording, so if sessions are
joined in time order, the test set is mostly the first session. The script warns when
fewer than 50 store frames reach test.

Alternative Considered: A random frame split (leaks).

## Fine-tune the store run from `best.pt` at a low learning rate

Date: on or before 2026-09-17 (`tools/train.py`)

Context: The model is already close; the store set will be small.

Decision: `store` starts from `runs/person/default/weights/best.pt` on
`data/merged/data.yaml` with `epochs=60`, `patience=15`, `lr0=0.001`, `warmup_epochs=1.0`.

Reason: Add one missing shape (seated, occluded people) without overwriting what the
model already knows. The full learning rate of 0.01 would wash it out.

Impact: The run has never been executed; it needs footage first.

Alternative Considered: Training from COCO again.

## Draft labels at conf 0.15

Date: on or before 2026-09-17

Context: Correcting drafts is faster than drawing boxes from nothing.

Decision: Drafts use `conf=0.15`, below the 0.25 operating point.

Reason: Favour recall, so that correcting is mostly deleting.

Impact: The drafts still miss exactly the seated and occluded people, so the operator
must add those. `PANDUAN.md` and the script's output both say so.

Alternative Considered: `--no-predraw`, which is supported.

---

**Stage: Repository and tooling**

## Commit code and metrics, not images or weights; keep `CLAUDE.md` local

Date: 2026-09-22 (initial commit `b3cef02`, which adds `.gitignore`)

Context: About 3 GB of data, weights and video, and a dataset licence that has not been
verified.

Decision: Commit `tools/`, the COCO JSON, manifest and YAML, the metrics JSON, figures and
training logs. Ignore images, weights, exports and video. Weights are to be published as
GitHub Release assets. `CLAUDE.md` and `.claude/` are ignored.

Reason: Size, and the unverified licence.

Impact: A fresh clone cannot train or run without the source export and the weights.
Derived images of the dataset **were** committed anyway (contact sheets, error
galleries, `rhc_preview.png`), which contradicts the stated reason; see
[memory.md](memory.md#known-stale-claims).

Alternative Considered: Git LFS (not used).

## Serve reports over HTTP on localhost, with Range support

Date: 2026-09-18 (`tools/serve_reports.py`); port-forwarding confirmed 2026-09-22

Context: The developer works through VSCode Remote-SSH from another machine. The built-in
video preview failed on both H.264 and VP8 files, so the problem was the remote transport
rather than the codec. Python's `http.server` answers every request with the whole file,
so browsers cannot seek.

Decision: `serve_reports.py` serves a folder with `206 Partial Content`, bound to
`127.0.0.1`, reached from the client through VSCode's port forwarding.

Reason: Seeking is needed to review tracking video. Binding to a network interface was
refused by the session's safety check and would expose the files; port forwarding
reaches only the developer's machine.

Impact: The port must be forwarded by hand (PORTS panel), because VSCode only
auto-detects servers started in its own terminal.

Alternative Considered: Copying the videos to the client; playing them in VLC on the
workstation.

## Backend comparison video: OpenVINO GPU in place of ONNX GPU

Date: 2026-09-22, decided with the user

Context: The user asked for side-by-side videos of PyTorch, ONNX and OpenVINO on CPU, and
PyTorch, ONNX and TensorRT on GPU.

Decision: The GPU video uses PyTorch, OpenVINO GPU and TensorRT. Both a frame-synced and
a real-time "race" version are rendered for each device, on the densest 10 s of
`data/video/people-walking.mp4` (frames 457-696, 8.44 people on average).

Reason: ONNX Runtime has no GPU provider here, and installing `onnxruntime-gpu` has broken
this Python before.

Impact: `tools/compare_backends_video.py` caches detections once per backend and renders
all four videos from the cache.

Alternative Considered: Installing `onnxruntime-gpu`; showing ONNX CPU inside the GPU video
under a CPU label.
