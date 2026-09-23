# Commands

Run everything from the repo root. Three scripts (`audit_duplicates.py`,
`clean_cctv.py`, `make_figures.py`) resolve paths against the working directory.

## Installation

**There is no dependency manifest** (no `requirements.txt` or `pyproject.toml`).
Packages are installed system-wide in `C:\Program Files\Python311` with no
virtualenv. These versions were read from that environment on 2026-09-23:

| Package | Version | Note |
|---|---|---|
| python | 3.11.9 | |
| torch, torchvision | 2.14.0+cu126, 0.29.0+cu126 | CUDA build; see below |
| ultralytics | 8.4.60 | ships YOLO26 |
| tensorrt | 11.3.0.99 | coexists with the cu126 torch |
| openvino | 2026.4.0 | |
| onnx, onnxruntime | 1.20.1, 1.21.1 | onnxruntime is CPU only |
| supervision, lap | 0.27.0, 0.5.13 | counting and tracking |
| pycocotools | 2.0.11 | |
| opencv-python, opencv-contrib-python | 4.10.0.84 | **plus** opencv-python-headless 4.13.0.90; see Debugging |
| numpy, pillow, matplotlib | 2.3.5, 11.3.0, 3.10.8 | |
| yt-dlp | 2026.8.19 | only used to fetch the test clip |

Install torch from the CUDA index first, then the rest. This command set is
reconstructed from the table above; it has not been run on a clean machine.

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install ultralytics==8.4.60 supervision==0.27.0 lap==0.5.13 pycocotools==2.0.11 \
            onnx==1.20.1 onnxruntime==1.21.1 openvino==2026.4.0 tensorrt==11.3.0.99
```

Check the channel before changing the CUDA version, because cu128 and cu129 carry
**older** torch builds than cu126:

```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cu126
```

Files the code needs that git does not contain:

| File | Where it comes from |
|---|---|
| `archieved_dataset/<Roboflow export>.coco.zip` | the Roboflow project export |
| `models/yolo26s.pt` (20.4 MB) | the official pretrained YOLO26s checkpoint |
| `runs/person/default/weights/best.pt` | meant to be a GitHub Release asset; whether one exists: Unknown - information not available in repository |

Restore the raw data. `unzip -d` does not create nested parents, so create the
folder first:

```bash
mkdir -p data/raw/cctv-person
unzip -q "archieved_dataset/CCTV-person-Forked on 9-16-2026.coco.zip" -d data/raw/cctv-person
```

## Development

Rebuild the dataset. Order matters, because `clean_cctv.py` reads the audit:

```bash
python tools/audit_duplicates.py   # -> reports/dup_audit.json  (about 3 min, reads 10k images)
python tools/clean_cctv.py         # -> data/cctv-person/
python tools/make_figures.py       # -> reports/figures/, reports/stats.json
```

Train and monitor:

```bash
python tools/train.py --list       # runs and the shared COMMON config
python tools/train.py smoke        # 2 epochs on 8% of the data: proves VRAM and pipeline
python tools/train.py all          # default, tuned, indoor (about 6 h in total)
python tools/train.py default tuned
python tools/status.py --watch     # refreshes every 30 s; shows default, tuned, indoor only
```

Evaluate:

```bash
python tools/evaluate.py --all                 # baseline, default, tuned, indoor on test
python tools/evaluate.py default --imgsz 416   # writes default_416.json
python tools/evaluate.py tuned --split val
python tools/make_eval_figures.py              # -> reports/eval/figures/
```

Count people in a video. Use `D:/...` paths, never `/d/...`:

```bash
python tools/count_people.py --source D:/ML/person-counter/data/video/people-walking.mp4 \
       --line 0.5,0,0.5,1 --name people-walking
python tools/count_people.py --source <video> --line 0,0.6,1,0.6 --max-frames 900
python tools/count_people.py --source <video> --no-video      # statistics only, faster
```

`--line` is `x1,y1,x2,y2` in 0-1. Draw it across the direction people walk: a
horizontal line for people moving up and down the frame, a vertical one for people
moving across.

Stage 6, once footage exists (see `PANDUAN.md`):

```bash
python tools/prepare_labeling.py --source D:/path/toko.mp4 --n 300   # -> data/store/toko/
#   correct the labels by hand in CVAT or labelImg, export YOLO into data/store/toko/labels/
python tools/merge_store_data.py --store data/store/toko             # -> data/merged/
python tools/train.py store                                          # fine-tune from best.pt
```

Evaluating the `store` run on store frames has no command yet; see
[features.md](features.md#store-frame-evaluation).

## Testing

**There is no automated test suite**: no `tests/` folder, no pytest configuration, no
linter configuration. Do not report that tests pass. What exists instead:

| Check | Command |
|---|---|
| The training pipeline runs and the batch fits in VRAM | `python tools/train.py smoke` |
| An ONNX export matches PyTorch | `python tools/export.py` (parity section of the output) |
| Every backend matches PyTorch, and their speed | `python tools/benchmark_backends.py` |
| The comparison-video tool runs end to end | `python tools/compare_backends_video.py --frames 12 --infer --render`. **This overwrites** the cache in `reports/eval/backend_compare/` and the four `backends_*.mp4`; run `--frames 240 --infer --render` afterwards to restore them |
| The dataset is consistent on disk | see [Database](#database) |

Parity output reports `match_pct` (how many reference boxes were found). Always
compare the box counts as well, since `match_pct` cannot see extra boxes.

## Database

There is no database. The equivalent tasks are about the data files.

Check that the cleaned dataset is consistent. Every number should be 4746, split
3796 / 475 / 475:

```bash
ls data/cctv-person/images | wc -l
tail -n +2 data/cctv-person/manifest.csv | wc -l
for s in train val test; do echo "$s $(ls data/cctv-person/yolo/images/$s | wc -l) $(ls data/cctv-person/yolo/labels/$s | wc -l)"; done
```

Before regenerating, clear the old output. Neither `clean_cctv.py` nor
`merge_store_data.py` deletes files, so a changed split leaves photos behind in their
old split folder:

```bash
rm -rf data/cctv-person/images data/cctv-person/yolo/images data/cctv-person/yolo/labels
rm -rf data/merged
```

## Build

"Build" here means exporting the model. Export **each size into its own folder**,
because ultralytics writes beside the weights and the last export overwrites the
previous one:

```bash
mkdir -p runs/person/<run>/trt640 && cp runs/person/<run>/weights/best.pt runs/person/<run>/trt640/
python -c "import os; os.environ['YOLO_AUTOINSTALL']='False'; from ultralytics import YOLO; \
YOLO('runs/person/<run>/trt640/best.pt').export(format='engine', imgsz=640, batch=1, half=False)"
```

This follows the existing layout (`runs/person/default/trt640/`, `ov640/`), which
`.gitignore` already excludes.

Swap in `format='onnx'` or `format='openvino'` for the other runtimes. An OpenVINO
export is a folder whose name must end in `_openvino_model`. A TensorRT build takes
about 2 minutes.

`python tools/benchmark_backends.py` exports all six variants itself, staging them in
`reports/eval/backends/`. As a side effect it also leaves the **last** size exported,
416, next to the weights.

Correct 640px exports that already exist:

| Runtime | Path |
|---|---|
| TensorRT | `reports/eval/backends/engine_640.engine`, `runs/person/default/trt640/best.engine` |
| ONNX | `reports/eval/backends/onnx_640.onnx` (opset 17), `runs/person/default/trt640/best.onnx` |
| OpenVINO | `reports/eval/backends/ov_640/best_openvino_model/`, `runs/person/default/ov640/best_openvino_model/` |

## Deployment

Deployment target, hosting, service manager and camera stream: Unknown - information
not available in repository. There is no deployment configuration, and
`count_people.py` only accepts a file path (it checks `os.path.exists`), so live
RTSP input is not supported.

The closest thing to a deployment command is running the counter with the chosen
backend:

```bash
# NVIDIA GPU: TensorRT
python tools/count_people.py --source <video> --weights reports/eval/backends/engine_640.engine
# CPU only: OpenVINO
YOLO_AUTOINSTALL=False python tools/count_people.py --source <video> \
       --weights reports/eval/backends/ov_640/best_openvino_model
```

`count_people.py` does not set `YOLO_AUTOINSTALL` itself, so set it in the shell
whenever the weights are not a `.pt` file. `count_people.py` does not pin an
OpenVINO device either, so this example could run on `AUTO`.

## Debugging

**Check an export's baked size before trusting it:**

```bash
python -c "import json;f=open('runs/person/default/weights/best.engine','rb');n=int.from_bytes(f.read(4),'little');print(json.loads(f.read(n))['imgsz'])"
python -c "import onnx;m=onnx.load('runs/person/default/weights/best.onnx');print([d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])"
grep -A2 imgsz runs/person/default/weights/best_openvino_model/metadata.yaml
```

All three print 416 for the files in `weights/` today.

**Is training still running?** `python tools/status.py`. Training detaches from the
session that started it; if a session ends, the process keeps going. Check the GPU
(`nvidia-smi`) before assuming a run has died.

**`torch.cuda.is_available()` is False:** a CPU wheel was installed over the CUDA one.
Re-run the torch install from [Installation](#installation).

**ONNX Runtime fails with `... has no attribute 'OrtCompileApiFlags'`:** ultralytics
tried to auto-install `onnxruntime-gpu` and half-succeeded.

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --no-cache-dir onnxruntime==1.21.1
```

**`import cv2` behaves oddly after a pip change:** three OpenCV wheels
(`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`) are installed and
share one `cv2` package. It currently imports as 4.10.0. If it breaks, uninstall all
three and install one.

**`cv2.VideoCapture(...).isOpened()` is False without an error:** the path is in Git
Bash form (`/d/...`). Use `D:/...`.

**A file count is short after `unzip`:** `unzip` silently skips entries whose path
would exceed 260 characters. Extract under a short base path, or read members with
Python's `zipfile`.

**Videos will not play in VSCode:** in a Remote-SSH session the built-in preview fails
for every codec. Serve the folder and forward the port:

```bash
python tools/serve_reports.py --dir reports/tracking --port 8910
# VSCode: PORTS panel -> Forward a Port -> 8910, then open http://localhost:8910/<page>.html
```

**A video from `count_people.py` will not play in a browser:** OpenCV writes `mpeg4`
(`mp4v`), which browsers cannot decode. Re-encode it to H.264. It is also written at an
integer fps (23 instead of 23.976), so it plays about 4% slow; `-r 24000/1001` before
`-i` fixes the timing without dropping frames:

```bash
ffmpeg -r 24000/1001 -i in.mp4 -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart \
       -fps_mode passthrough out_h264.mp4
```

**`WARNING NMS time limit ... exceeded`** appeared once while tracking with a TensorRT
engine, even though the model is NMS-free. It was not investigated.
