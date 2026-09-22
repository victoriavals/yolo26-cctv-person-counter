# yolo26-cctv-person-counter

Fine-tuning YOLO26s into a single-class `person` detector for **indoor store CCTV**
(ceiling-mounted camera), plus the tooling around it: dataset cleaning, evaluation,
ONNX/OpenVINO/TensorRT export with parity checks, and a ByteTrack-based line counter.

## Read this first: the model does not work on real ceiling CCTV yet

The detector reaches **mAP50 0.891** on a clean, leak-free held-out test split. That
number did **not** predict field behaviour.

Run over 900 frames of a 1920x1080 warehouse packing-room CCTV feed (ceiling camera,
~35 degrees, people working among shelves -- the closest domain match available):

- reported occupancy averaged **2.35 people against 5-7 visibly present**, so roughly
  **60-80% of people were missed**;
- the tracker emitted **40 unique IDs for about 6 people** -- not 40 people, but tracks
  repeatedly lost and re-acquired because detection is intermittent.

The cause was measured, not guessed. Detections were counted on the same 6 frames while
sweeping both knobs:

| imgsz | conf 0.25 | conf 0.15 | conf 0.08 |
|-------|-----------|-----------|-----------|
| 640 | 15 | 17 | 21 |
| 960 | 17 | 19 | 19 |
| 1280 | 16 | 17 | 19 |
| 1600 | 15 | 17 | 22 |

Going from 640 to 1600 -- 6x the pixels -- moves nothing (15 -> 15). So it is neither
resolution nor threshold: **the model does not recognise seated, table-occluded people.**
Training boxes run h/w ~2.1 (standing or walking); a seated worker behind a table is a
shape it has never seen, and no inference setting teaches it one.

What follows from this:

- Do not treat the numbers in `reports/eval/` as deployment estimates.
- Any labelled footage you add **must** include seated, crouching and half-occluded
  people. That is the specific gap.
- Do not tune the tracker yet. 40 IDs for 6 people is a symptom of broken detection, not
  of tracker settings.

## What is and is not in this repository

Committed: all code (`tools/`), COCO ground truth and the split manifest, evaluation
metrics as JSON, every figure, and per-run training logs (`results.csv`, `args.yaml`,
loss and PR curves).

Not committed, because it totals ~3 GB: the images themselves, the source Roboflow
archive, rendered tracking video, and all model weights and compiled graphs
(`.pt` / `.onnx` / `.engine` / OpenVINO IR). Weights are published as GitHub Release
assets instead. See [Rebuilding the dataset](#rebuilding-the-dataset) to regenerate
`data/` from the source export.

```
data/cctv-person/
  annotations/{train,val,test}.json   COCO, single class `person` (id 1)   [committed]
  manifest.csv                        per photo: scene, split, copies      [committed]
  yolo/data.yaml                      Ultralytics config                   [committed]
  images/, yolo/{images,labels}/      4886 photos                          [not committed]
models/yolo26s.pt                     pretrained starting weights          [not committed]
tools/                                the whole pipeline
reports/                              metrics, figures, HTML reports
runs/person/<name>/                   one directory per experiment
PANDUAN.md                            Indonesian field guide for collecting your own footage
```

## Results on the test split

475 images, 1844 boxes. `MAE` is mean absolute error in people per frame, `bias` the
signed mean error, `+-1` the share of frames counted within one person.

| model | conf | mAP50 | mAP50-95 | MAE | bias | +-1 | P | R |
|-------|------|-------|----------|-----|------|-----|---|---|
| baseline (COCO, no fine-tune) | 0.15 | 0.644 | 0.288 | 1.600 | -0.94 | 65.9% | 0.776 | 0.588 |
| **default** | 0.25 | **0.891** | **0.584** | **0.640** | -0.12 | **88.0%** | **0.873** | **0.845** |
| tuned | 0.20 | 0.873 | 0.581 | 0.659 | -0.10 | 85.7% | 0.853 | 0.832 |
| indoor-only | 0.15 | 0.679 | 0.341 | 1.322 | -0.86 | 68.2% | 0.800 | 0.622 |

Use `runs/person/default/weights/best.pt` at `conf=0.25`.

**Where the gain came from.** The COCO baseline was weakest in exactly the deployment
domain: indoor mAP50 0.551 against outdoor 0.792. After fine-tuning the gap closes to
0.904 vs 0.924, and indoor recall goes 0.489 -> 0.859. The scatter plot in
`reports/eval/figures/scatter.png` shows the baseline flat-lining around 10 detections on
frames holding 35+ people, while `default` tracks the diagonal.

**Two hypotheses were tested and both failed. Do not retry them without new evidence:**

1. *Reduce augmentation, since the pixels are already augmented and the camera is fixed.*
   The `tuned` run lowered `hsv_v`, `hsv_s` and `scale`, and came out slightly worse on
   nearly everything. Ultralytics defaults are good enough for this data.
2. *Train on indoor only, in case outdoor data pulls the model away from the store domain.*
   The `indoor` run reached mAP50 0.885 on the indoor subset -- still **below** `default`'s
   0.904, which also trained on outdoor -- and collapsed to 0.270 on `unknown` scenes.
   Outdoor data **helps** indoor performance.

Remaining weaknesses: AP_small 0.283 (distant people; was 0.061), MAE 0.97 in the 6-10
person bucket, mAP50 0.789 on `unknown` scenes, and a small persistent undercount bias
of -0.12.

Note that YOLO26s is **end-to-end / NMS-free** (`end2end=True`, `reg_max=1`, 10.0M
params, 22.8 GFLOPs). Threshold tuning is therefore `conf` only -- there is no NMS `iou`
knob to sweep.

## Deployment: pick TensorRT on GPU, OpenVINO on CPU. Never ONNX.

All six backends are exactly faithful to PyTorch (155/155 boxes at IoU>=0.95, identical
counts on 40/40 images), so this is purely a speed decision. Measured on an i5-13400F CPU
and an RTX 4060 Ti GPU; raw numbers in `reports/eval/backends_default.json`.

| backend | HW | 640px | FPS | 416px | FPS | parity | max dconf |
|---------|----|-------|-----|-------|-----|--------|-----------|
| PyTorch | CPU | 257.4 ms | 3.9 | 128.4 ms | 7.8 | (reference) | -- |
| ONNX | CPU | 696.2 ms | 1.4 | 398.9 ms | 2.5 | 100% | 0.00021 |
| **OpenVINO** | CPU | **50.2 ms** | **19.9** | **27.5 ms** | **36.4** | 100% | 0.00030 |
| PyTorch | GPU | 26.8 ms | 37.3 | 24.9 ms | 40.2 | 100% | 0.07932 |
| OpenVINO | GPU | 37.4 ms | 26.8 | 35.4 ms | 28.2 | 100% | 0.00032 |
| **TensorRT** | GPU | **12.8 ms** | **78.3** | **11.0 ms** | **90.8** | 100% | 0.09805 |

- **GPU: TensorRT**, 2.1x faster than PyTorch on the same card. Exported fp32 so parity
  stays comparable; fp16 would be faster but is a different model.
- **CPU: OpenVINO**, 5.1x faster than PyTorch on the same CPU. 19.9 FPS at full 640px
  already clears a 15 fps camera.
- **ONNX is the worst option**, 14x slower than OpenVINO on CPU, consistently at both
  sizes. opset 17 recovers only 7%.
- **OpenVINO on an NVIDIA GPU is pointless**: slower than PyTorch GPU, and at 416px
  slower than OpenVINO on the CPU. Its GPU plugin targets Intel hardware.

Parity calibration: PyTorch GPU-vs-CPU differs by 0.079 and TensorRT by 0.098, far more
than OpenVINO's or ONNX's 0.0003. That is ordinary fp32 kernel spread, not lost accuracy
-- box counts are identical. Treat ~0.1 as the noise floor when judging a backend, not 0.

### Speed and accuracy by inference size

| imgsz | CPU ms | CPU FPS | GPU FPS | mAP50 | AP_small | MAE | indoor MAE | best conf |
|-------|--------|---------|---------|-------|----------|-----|------------|-----------|
| 640 | 260 | 3.8 | 37.4 | 0.891 | 0.283 | 0.640 | 0.562 | 0.25 |
| 512 | 177 | 5.6 | -- | 0.878 | 0.267 | 0.686 | 0.580 | 0.25 |
| **416** | 128 | **7.8** | 40.1 | 0.869 | 0.251 | 0.686 | **0.523** | 0.20 |
| 320 | 101 | 9.9 | -- | 0.822 | 0.173 | 0.842 | 0.659 | 0.15 |

The CPU column is PyTorch. With OpenVINO (19.9 FPS) or TensorRT (78.3 FPS), 640px is far
above what a 15 fps camera needs, so dropping resolution is unnecessary. 416 remains
useful if a higher frame rate is wanted, and it is not a compromise indoors -- indoor MAE
actually improves (0.562 -> 0.523). 320 is the cliff: AP_small collapses.

**The best `conf` shifts with `imgsz`, so re-tune it whenever `imgsz` changes.** It is not
inheritable. On GPU, 640 and 416 cost the same, so keep 640 there.

### Export traps, all hit in practice

- Ultralytics writes exports **beside the weights**, so exporting a second size overwrites
  the first -- and if that model is still loaded, the file is locked
  (`RuntimeError: Can't open bin file`). Stage each export into its own directory before
  loading anything.
- An OpenVINO model is identified by its **directory name**, which must end in
  `_openvino_model`. Any other name fails with `is not a supported model format`.
- The exported graph bakes `imgsz` in, so benchmarking another resolution needs a fresh
  export. Reusing the 640 graph at 416 silently keeps running at 640.
- **Pin the device explicitly.** Ultralytics falls back to OpenVINO's `AUTO` whenever
  anything besides CPU is enumerated -- and OpenVINO lists an NVIDIA card as `GPU` -- so
  `device="cpu"` does **not** guarantee a CPU measurement. Use `intel:cpu` / `intel:gpu`.
- The ONNX export **fixes batch at 1**. Feeding it more raises
  `Got invalid dimensions for input: images`.
- Loading an `.onnx` model while CUDA is present makes Ultralytics try
  `pip install onnxruntime-gpu`, which can break a working onnxruntime mid-install.
  Set `os.environ["YOLO_AUTOINSTALL"] = "False"` **before** importing ultralytics, as
  `tools/export.py` does.

## The source dataset was broken in four ways

The data comes from a Roboflow export of a CCTV person project. All four defects are
invisible from the folder structure and are contradicted by the bundled README. Read this
before going back to a raw export.

**1. The `categories` table is junk from a dataset merge.** It lists
`0 person, 1 "0", 2 "CCTV persons - v1 2024-09-16 8:18pm", 3 Person, 4 no_safetyshoes, 5 safety_shoes`.
Category 0 -- the only sensibly named one -- carries **zero** annotations; 99.5% of boxes
sit on `category_id: 2`, the timestamp-named entry. Code keying off `name == "person"`
silently gets nothing.

**2. Two of those categories are shoes, not people, and they appear only in `valid`.**
`no_safetyshoes` (39) and `safety_shoes` (21) are 60 boxes with h/w 1.0-1.4 covering 0.3%
of the frame; person boxes here run h/w 2.1. `clean_cctv.py` keeps categories 0-3 as
`person` and drops 4-5. **Check all three splits before concluding anything about the
class distribution** -- train and test are pure `category_id: 2`, so a train-only check
misses this entirely.

**3. Half the export is augmented duplicates.** 10062 files are 4886 distinct photographs.
Augmentation (rotation with corner fill, noise, exposure shift) is baked into the pixels,
despite the export's README claiming none was applied -- that line describes the fork's
export step, not the upstream images.

**4. The exported split leaks.** 1651 photographs have copies in more than one split:
**75% of test files and 70% of valid files** share a photograph with another split. Any
mAP computed on the original test split is meaningless.

**Duplicates must be found by pixel content, not by filename.** Roboflow keeps the
original stem, and 1507 stems hold more than one *distinct* photograph -- 44 files named
`youtube-0_jpg.rf.<hash>.jpg` are not one photo. Grouping by name discards real images.
`tools/audit_duplicates.py` clusters on a rotation-tolerant HSV+gray histogram over
non-fill pixels at threshold 0.97 (within-stem median similarity 0.989 vs 0.514 across
stems; 0.30% false-positive rate on unrelated pairs).

When comparing two Roboflow exports of the same project, the `.rf.<hash>` suffix is
re-rolled per export, so filenames never match. Compare by stem:
`re.sub(r"\.rf\.[A-Za-z0-9]+\.jpg$", "", name)`.

## The cleaned dataset

4886 unique photos, single class `person`, split **80/10/10 and stratified**.

| split | images | boxes | boxes/img | indoor share |
|-------|--------|-------|-----------|--------------|
| train | 3796 | 14230 | 3.75 | 37.0% |
| val | 475 | 2371 | 4.99 | 37.1% |
| test | 475 | 1844 | 3.88 | 37.1% |
| total | 4746 | 18445 | 3.89 | 37.0% |

1591 source scenes, none spanning two splits. No empty images. The YOLO labels were
verified box-for-box against the COCO JSONs.

**Why stratified.** A plain global greedy split keeps 80/10/10 but lets composition drift:
an earlier attempt landed **18% indoor in test** against 37% overall, and a test median of
5 people per frame against train's 2 -- it measured an indoor-store model mostly on
outdoor crowds. `assign_splits()` allocates inside `(domain kind x density bucket)`
strata, carrying the split deficit across strata so the global ratio stays exact. Scenes
still move as whole units, so the no-leakage guarantee is unchanged.

### Domain composition

Families are grouped from source filename patterns, each then checked visually against
random contact sheets (`reports/figures/sheet_cctv_*.png`).

| kind | photos | share | families |
|------|--------|-------|----------|
| CCTV indoor | 1755 | 37.0% | `01-08-2022__*` shops/warehouses/offices, `image_*` corridors, `opencv_frame_*` labs, `output_mp4*` lobbies |
| CCTV outdoor | 1489 | 31.4% | `Day_*` streets, `SeieeDept-*` campus, `CrossRoad/Road/Bridge*`, `scene*` |
| unidentified | 1360 | 28.7% | mixed; filename has no usable pattern |
| entrance / semi | 142 | 3.0% | `0408xx_*` |
| ~~not CCTV~~ | ~~140~~ | dropped | `youtube-*` -- webcams and selfies, wrong angle and lens |

`tools/families.py` is the single definition of these patterns and of the
`indoor / outdoor / semi / bukan-cctv / unknown` kinds; `clean_cctv.py` and
`make_figures.py` both import it. `DROP_KINDS` there controls what is dropped at load time.

### Known limitations of this data

- **Density is the ceiling on what it can teach.** 59% of frames hold fewer than 4 people
  and only 1.1% hold more than 12. A model trained on this alone degrades when the store
  gets busy, and nothing in this repo fixes that. `mosaic` is deliberately left at 1.0 in
  training: compositing 4 images per sample is the only thing offsetting this.
- **No steep top-down views and no strong fisheye.** If your camera points almost straight
  down, your own labelled footage is mandatory, not optional.
- **No track IDs**, so the association half of a tracker has no training or evaluation data
  here. Tracker quality has to be measured on your own footage.
- **The `semi` family (entrances, 142 photos, 3.0%) lands entirely in `train`**, so
  entrances are never measured on the held-out splits. This is deliberate. Forcing
  representation by reserving each stratum's smallest scenes was tried and made things far
  worse: test swung to 60.9% outdoor and 1.3% unknown. A balanced indoor share and an exact
  80/10/10 were judged worth more than covering a 3% family.
- MOT17 was evaluated as a source of track IDs and **dropped**: 7 scenes, 6 outdoor street
  plus 1 indoor mall shot from a moving head-height camera, so its perspective geometry
  does not match a ceiling-mounted store camera.

## Rebuilding the dataset

Put the Roboflow COCO export in `archieved_dataset/`, extract it, then run the pipeline in
order -- `clean_cctv.py` consumes the audit's cluster assignments.

```bash
# unzip -d does NOT create nested parents, so mkdir first
mkdir -p data/raw/cctv-person
unzip -q "archieved_dataset/<export>.coco.zip" -d data/raw/cctv-person

python tools/audit_duplicates.py   # -> reports/dup_audit.json  (~3 min, reads 10k images)
python tools/clean_cctv.py         # -> data/cctv-person/
python tools/make_figures.py       # -> reports/figures/, reports/stats.json
```

`data/cctv-person/yolo/data.yaml` carries an **absolute** `path:`. Update it if the project
moves.

To change only the split, `clean_cctv.py` alone is enough -- `reports/dup_audit.json` is
committed, so the audit does not need re-running. But the raw images must be back on disk
first, so the `unzip` is always step one.

## Training, evaluation and deployment

```bash
python tools/train.py --list          # show runs and the shared config
python tools/train.py smoke           # 2 epochs on 8% -- proves VRAM and pipeline
python tools/train.py all             # default, tuned, indoor  (~6 h total)
python tools/status.py [--watch]      # progress of every run

python tools/evaluate.py --all        # test-split metrics -> reports/eval/*.json
python tools/evaluate.py default --imgsz 416   # suffixes output as default_416.json
python tools/make_eval_figures.py     # -> reports/eval/figures/

python tools/export.py                # ONNX + parity + speed -> reports/eval/
python tools/benchmark_backends.py    # 6 backends, CPU+GPU, + parity
python tools/make_backend_figure.py   # -> reports/eval/figures/backends.png

python tools/count_people.py --source <video> --line 0,0.6,1,0.6 --max-frames 900
```

`tools/train.py` holds every experiment as one entry in `RUNS`, sharing a `COMMON` config
so differences can only come from the declared overrides. The augmentation reasoning for
the `tuned` run is written out in that file.

## Adding your own store footage

Because the test-split metrics do not predict field behaviour, footage from the actual
deployment camera is a **requirement**, not a nice-to-have.

```bash
python tools/prepare_labeling.py --source toko.mp4 --n 300   # -> data/store/<name>/
# ... label in CVAT / Label Studio / Roboflow, export YOLO ...
python tools/merge_store_data.py --store data/store/toko     # -> data/merged/
```

**[`PANDUAN.md`](PANDUAN.md) is the step-by-step operator guide for this stage** (in
Indonesian): how much footage to record and when, what must appear in it, which labelling
tool, the labelling rules, and the pass/fail thresholds for deciding whether the result is
deployable.

`prepare_labeling.py` samples frames worth labelling and pre-annotates them: a minimum
time gap between frames, frames where nothing moved dropped, crowd-bucket stratification
(15/30/30/25 across 0, 1-2, 3-5, 6+ people, because a store is quiet most of the time and
uniform sampling would spend the whole budget on empty frames), and draft boxes at
`conf=0.15` -- below the 0.25 operating point, so correcting is mostly deleting.

**Do not reuse the HSV descriptor from `audit_duplicates.py` for a fixed camera.** That one
separates different *scenes*; on a static CCTV the histogram is dominated by the unchanging
background, so people moving barely shift it. Measured on the warehouse clip, it dropped 88
of 99 candidates, leaving 11 frames out of 60 requested. `prepare_labeling.py` compares
downscaled grayscale pixels instead (`--min-change`, default 2.0 mean absolute difference),
which registers movement. With that, all 99 candidates passed.

`merge_store_data.py` splits store frames by **contiguous time block**, not at random --
frames seconds apart look alike, and a random split would leak them across train and test,
the exact defect that made the original Roboflow split meaningless. It writes two configs:

- `data/merged/data.yaml` -- base dataset plus store frames, for fine-tuning;
- `data/merged/data_store_test.yaml` -- val and test are **store frames only**. This is the
  number that predicts field behaviour. Test share defaults to 20%, higher than the usual
  10%, and the script warns below ~50 store test frames.

Fine-tune from `runs/person/default/weights/best.pt`, not from COCO again.

## Environment

Verified, not assumed.

| | |
|---|---|
| GPU | RTX 4060 Ti, 8 GB VRAM, driver 591.86 (CUDA 13.1 capable) |
| torch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| ultralytics | 8.4.60 (ships YOLO26 at `cfg/models/26/yolo26.yaml`) |
| model | `models/yolo26s.pt`, 20.4 MB |
| also | `lap` 0.5.13 (ByteTrack/BoT-SORT), supervision 0.27.0, onnx, onnxruntime 1.21.1, pycocotools, openvino 2026.4.0, tensorrt 11.3.0.99 |
| RAM | 34 GB -- `cache=True` holds the whole 300 MB image set |

If `torch.cuda.is_available()` returns False, a CPU wheel has been installed over the CUDA
one. Fix with
`pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision`. Check
channel contents first --
`pip index versions torch --index-url https://download.pytorch.org/whl/<cuXXX>` -- because
cu128 and cu129 carry *older* torch than cu126 does.

### Windows notes

- **Ultralytics ignores a relative `project=`.** It resolves one against its own global
  `runs_dir` in `%APPDATA%\Ultralytics\settings.json`, so runs land somewhere unrelated.
  `train.py` passes an absolute path; `tools/status.py` checks both locations.
- **Keep dataloader workers low.** With the default `workers=8`, training holds a train
  loader *and* a val loader alive, so 16 worker processes plus pin-memory threads exist at
  once and validation dies with `RuntimeError: Pin memory thread exited unexpectedly`.
  Probed in isolation, `val()` survives 0/2/4/8 equally -- the count alone is not the fault,
  it is the two loaders together. `train.py` pins `workers=2`. Images are cached in RAM, so
  workers only do augmentation.
- **`model.predict()` on a list of paths renames the results.** `r.path` comes back as
  `image0.jpg`, `image1.jpg`, ... not the input path, so any dict keyed on
  `basename(r.path)` silently fails to join against ground truth. It also loads the whole
  list as one batch -- 475 test images asks for 6.2 GB and OOMs an 8 GB card.
  `evaluate.py` feeds paths in chunks of 16 and takes filenames from the input order.
- **OpenCV does not understand Git Bash paths.** `cv2.VideoCapture("/d/ML/...")` fails
  silently (`isOpened()` is False); use `D:/ML/...`.
- The filesystem is case-insensitive, and four source images differ from another only by
  case (`1_PNG_jpg` vs `1_png_jpg`), silently overwriting each other until `clean_cctv.py`
  started disambiguating with a `__2` suffix. Check `ls | wc -l` against the manifest row
  count after any bulk copy.
- **`unzip` silently skips entries whose destination path would exceed 260 characters, and
  reports success.** Extracting this dataset under a 123-character base loses 20 files;
  under a 47-character base it loses none. When a file count comes up short after
  extraction, suspect this before suspecting the archive -- or read members straight from
  the zip with Python's `zipfile`, which has no such limit.
- Python's default stdout encoding is cp1252, so printing non-latin-1 characters raises
  `UnicodeEncodeError`. Script output is kept ASCII.

## Licence

The code in `tools/` is offered as-is for reference. **The licence of the upstream Roboflow
CCTV person dataset has not been verified**, which is why no images are committed here and
why you should check the source project's terms before redistributing any of the data or a
model trained on it.
