# Conventions

These are the patterns the existing code follows. Match them in new code.

## Language

| Where | Language |
|---|---|
| Code, identifiers, comments, docstrings | English |
| Console output (`print`), argparse help, `sys.exit` messages | **Indonesian** |
| Output field names in tracking files (`t_detik`, `masuk`, `keluar`) and some labels (`grid_*_baik.png`, family labels in `families.py`) | Indonesian |
| `README.md`, `CLAUDE.md`, `.agents/` | English |
| `PANDUAN.md`, the HTML report pages | Indonesian |

**Console output must be ASCII.** Python's stdout here is cp1252, so characters
such as `∩` raise `UnicodeEncodeError`. Write `+-1`, `->`, `x`.

## Naming

| Thing | Rule | Examples |
|---|---|---|
| Scripts | `snake_case`, usually verb_noun | `audit_duplicates.py`, `prepare_labeling.py`, `count_people.py`; single nouns for core steps: `train.py`, `evaluate.py`, `export.py` |
| Module constants | `UPPER_CASE` at the top, each with a comment giving the reason | `CONF = 0.25`, `SIM_THRESHOLD = 0.97`, `MIN_CHANGE = 2.0` |
| Run names | lower-case single word, a key in `RUNS` | `smoke`, `default`, `tuned`, `indoor`, `store` |
| Eval outputs | `<run><suffix>.json` and `<run><suffix>_preds.npz`; the suffix is `_<imgsz>` when not 640 | `default_416.json` |
| Benchmark outputs | `backends_<run>.json`, `export_<run>.json` | |
| Staged exports | `<fmt>_<size>.<ext>`, `ov_<size>/best_openvino_model/` | `engine_640.engine` |
| Tracking outputs | `<name>_annotated.mp4`, `_counts.csv`, `_summary.json`; `<name>` defaults to the video's stem | `warehouse_summary.json` |
| Store frames | `<name>_<frame index, 7 digits>.jpg` | `toko_0001234.jpg` |
| OpenVINO models | the directory **must** end in `_openvino_model` | |

## Folders

| Folder | Holds | In git |
|---|---|---|
| `tools/` | every script | yes |
| `data/cctv-person/` | the cleaned dataset | JSON, CSV and YAML only |
| `data/raw/`, `data/store/`, `data/merged/`, `data/video/` | raw export, store frames, merged set, test clips | no |
| `models/` | pretrained starting weights | no |
| `runs/person/<run>/` | training output | logs and plots yes; `weights/` and `trt*/` no |
| `reports/` | metrics JSON, figures, HTML pages | yes, except videos and `reports/eval/backends/` |
| `archieved_dataset/` | the source zip (the spelling is intentional; keep it) | no |

Put a new output next to its siblings: metrics in `reports/eval/`, figures in
`reports/eval/figures/` or `reports/figures/`, video in `reports/tracking/`.

## Script structure

Every script follows this shape:

```python
"""One line saying what it does.

Why it exists and what would go wrong without it, with measured numbers.

    python tools/<script>.py            # usage examples
    python tools/<script>.py --flag
"""
import argparse
import os
import sys

os.environ["YOLO_AUTOINSTALL"] = "False"   # before ultralytics, whenever exports may load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONST = 0.25          # why this value, and what measurement chose it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", help="bantuan dalam bahasa Indonesia")
    a = ap.parse_args()
    from ultralytics import YOLO    # heavy import inside main(), so --help stays fast
    ...


if __name__ == "__main__":
    main()
```

Details that recur:

- **Paths.** Build from `ROOT`, and call `.replace("\\", "/")` before handing a path to
  ultralytics or OpenCV. OpenCV fails silently on Git Bash paths such as `/d/ML/...`.
  Three older scripts (`audit_duplicates.py`, `clean_cctv.py`, `make_figures.py`) use
  paths relative to the working directory instead, so run them from the repo root.
  New code should use `ROOT`.
- **Shared code.** Import sibling modules with
  `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` followed by
  `from families import ...  # noqa: E402`.
- **Lazy heavy imports.** Import `ultralytics` (and `torch`) inside functions.
- **Docstrings on helpers** describe the return value with an arrow:
  `"""-> (label, kind)."""`.
- **No type hints** anywhere. Do not add them to one file alone.
- Metric and summary JSON is written with `indent=1` so it diffs well. Large or
  machine-only JSON (COCO annotations, `dup_audit.json`, caches) is written without
  indentation. CSV is written with `csv.DictWriter`.
- Figures use matplotlib with the `Agg` backend. Contact sheets use PIL with fonts
  loaded from `C:/Windows/Fonts`, falling back to the default font.

## Comments

Comments explain **why**, and usually quote the measurement that decided it:

```python
patience=30,        # ultralytics default is 100, i.e. effectively no early stop
```

When an alternative was tried and rejected, the comment says so and gives the
result (see the `semi` note in `clean_cctv.py` or the augmentation block in
`train.py`). Keep doing this; it is how the next agent avoids repeating a failed
experiment.

## Error handling

- **Fail fast, with a fix.** Missing input ends the script with `sys.exit()` and an
  Indonesian message saying what to run, for example
  `"... belum ada. Jalankan dulu: python tools/merge_store_data.py ..."`.
- **Isolate failures in batch measurements.** `benchmark_backends.py` wraps every
  backend and every size in its own `try/except`, so one failure does not throw away
  results that already succeeded. Record the error in the output JSON instead.
- **Warn instead of stopping when the data is merely thin**, for example "fewer than
  50 store test frames" or "only N frames passed".
- Do not catch exceptions just to hide them. The only silent `except` blocks are for
  optional inputs such as fonts or the ultralytics settings file.

## Validation

- **Print the calibration next to the threshold.** `audit_duplicates.py` prints the
  within-stem and across-stem similarity so that the 0.97 threshold can be checked.
- **Check parity for every export.** Compare boxes against PyTorch at IoU >= 0.95, and
  compare box **counts** as well: `match_pct` alone cannot see extra boxes.
- **Verify labels after any conversion.** The YOLO labels were checked box by box
  against the COCO JSON.
- **Compare file counts** with the manifest after bulk copies or extraction, because
  of case collisions and `unzip`'s silent 260-character limit.

## Security and privacy

- CCTV frames show identifiable customers and staff. Do not upload them to a
  third-party service (Roboflow included) without the user's explicit decision.
- The licence of the upstream dataset has not been verified. Do not commit new images
  derived from it or from other third-party footage, such as the RHC zip.
- Set `YOLO_AUTOINSTALL=False` before importing ultralytics in any script that can load
  a non-PyTorch model. **`count_people.py` does not do this yet** even though
  `--weights` accepts an `.onnx` path, so set the variable in the shell when using it.
- Bind servers to `127.0.0.1`. `serve_reports.py` does so by default; reach it from
  another machine through port forwarding, not by binding a network interface.
- Keep machine-specific network details, user names and e-mail addresses out of
  committed files.

## Performance

- Exported models take **one frame per call**. ONNX fixes batch at 1, and that is how
  production runs anyway.
- `model.predict()` on a list loads the whole list as one batch. Chunk it; 16 images
  per chunk fits in 8 GB.
- Keep dataloader `workers=2` on Windows (see [decisions.md](decisions.md)).
- `cache=True` for training: the images take 300 MB against 34 GB of RAM.
- Warm up before timing (the scripts use 5 to 8 frames), and call
  `torch.cuda.synchronize()` around GPU timings.
- Export fp32. fp16 would be faster but is a different model and breaks parity
  comparison.
- One export per `(format, imgsz)`, each in its own directory.

## Before finishing a change

- Run the script on a small input first: `train.py smoke`, or
  `compare_backends_video.py --frames 12`.
- If you changed a number that a document quotes, update `README.md`, `PANDUAN.md`
  and `.agents/` together.
- There is no linter or test suite in the repo. Do not claim that tests passed.
