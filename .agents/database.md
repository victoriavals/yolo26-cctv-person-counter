# Data layer

## Database overview

There is **no database**: no engine, schema migrations, ORM or query layer.
Database engine: Unknown - information not available in repository, because none
is used. All persistent state is files. This document treats those files as the
"tables" an agent reads and writes, and records their exact fields.

Scale, measured on 2026-09-23:

| | |
|---|---|
| Source export | 10062 files holding 4886 distinct photos |
| Cleaned dataset | 4746 photos (140 `youtube-*` dropped), 18445 boxes, one class, 1591 scenes |
| Splits | train 3796 / val 475 / test 475 images, all scene-disjoint |
| Image size | 640x640 (the export's size) |

## Main tables

| Table (file) | Purpose | Relationship | Important fields |
|---|---|---|---|
| `data/cctv-person/annotations/{train,val,test}.json` | COCO ground truth, committed | `images[].file_name` = `manifest.file`. **Image IDs restart at 1 in every split** | `images[id, file_name, width, height]`, `annotations[id, image_id, category_id=1, bbox, area, iscrowd=0]`, `categories=[{id:1, name:"person"}]` |
| `data/cctv-person/manifest.csv` | One row per cleaned photo, committed | `file` joins the COCO JSON and the YOLO images; `scene` groups photos | `file, scene, split, kind, density_bucket, boxes, copies_in_export, split_in_export` |
| `data/cctv-person/yolo/labels/{split}/<stem>.txt` | Ultralytics labels | `<stem>` = image name without `.jpg` | one line per box, `0 cx cy w h`, normalised, 6 decimals. An empty file means no people |
| `data/cctv-person/yolo/data.yaml` | Training dataset config, committed | points at `images/{split}` | **absolute** `path:`, `train/val/test`, `nc: 1`, `names: [person]` |
| `yolo/data_indoor.yaml`, `yolo/train_indoor.txt` | Config for the `indoor` run | train is the indoor list; val and test are the same as `data.yaml` | written by `train.py` whenever `indoor` is run |
| `reports/dup_audit.json` | Duplicate clusters, committed | `assignments[raw file_name] -> cluster ID`, read by `clean_cctv.py` | `files, stems, stems_multi_photo, distinct_photos, photos_with_copies, photos_leaking, files_in_leaking_photos, threshold, assignments` |
| `reports/stats.json` | Chart data for the dataset report | derived from the manifest and the audit | `images, boxes, density_mean, density_hist, empty_images, boxh_hist, boxh_median_px, splits, boxes_per_split, scenes, copies_hist, audit, families, domain_kind` |
| `runs/person/<run>/results.csv` | Per-epoch training log (ultralytics) | read by `status.py` and `make_eval_figures.py` | `epoch, time, train/*_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)`, ... |
| `runs/person/<run>/args.yaml` | The full resolved training config | | every ultralytics argument |
| `runs/person/<run>/run_meta.json` | "Training finished" marker | read by `status.py` | `run, seconds, overrides, model, weights` |
| `reports/eval/<run>[_<imgsz>].json` | Test-split metrics | `per_image[].file` = COCO `file_name` | `name, weights, split, imgsz*, model_classes, images, conf_best_counting, conf_best_f1, overall{}, at_best_counting{}, at_best_f1{}, sweep[], by_kind{}, by_density{}, per_image[]` |
| `reports/eval/<run>[_<imgsz>]_preds.npz` | Raw predictions, so figures need no second inference | key = image `file_name` | `float32 [N, 5]`: x, y, w, h (pixels), score. Every box with score >= 0.001 |
| `reports/eval/export_<run>.json` | ONNX parity and speed | | `run, weights, conf, n_images, onnx, onnx_mb, parity{}, speed{pytorch_gpu, pytorch_cpu, onnx_cpu}` |
| `reports/eval/backends_<run>.json` | Six-backend benchmark | `artifacts` lists the staged exports | `run, conf, n_images, sizes, gpu, hw_of, backends{label:{size: ms}}, parity{label:{ref_boxes, boxes, matched, match_pct, max_conf_delta, count_equal_images}}, artifacts` |
| `reports/eval/backend_compare/<backend>.json` | Per-frame cache behind the comparison videos | frame `i` = frame `457 + i` of `data/video/people-walking.mp4` | `key, label, device, weights, latency[], boxes[][] (xyxy px), ms_mean, ms_median, fps, ms_inference_only, total_boxes` |
| `reports/tracking/<name>_counts.csv` | Per-frame counting trace | | `frame, t_detik, occupancy, masuk, keluar, id_kumulatif` |
| `reports/tracking/<name>_summary.json` | Summary of one counting run | | `source, weights, tracker, conf, frames, fps_video, detik_diproses, waktu_proses_detik, fps_proses, occupancy{mean, median, max, min}, track_unik, garis_masuk, garis_keluar, garis` |
| `data/store/<name>/manifest.csv` | Frames sampled for labelling | `file` joins `images/` and `labels/` | `file, frame, detik, draft_boxes` |
| `data/merged/store_manifest.csv` | Store frame to split | | `file, split` |
| `data/merged/test_store.txt` | Store test images, one absolute path per line | read through `data_store_test.yaml` | |

`*` The `imgsz` key exists only in files written after it was added to
`evaluate.py`. `reports/eval/default.json` does not have it; treat a missing value
as 640.

Field names in the tracking outputs are Indonesian: `t_detik` is time in seconds,
`masuk`/`keluar` are line crossings in and out, and `id_kumulatif` is the number of
unique track IDs so far.

## Entity relationships

```
raw export file (<stem>.rf.<hash>.jpg)
      | N:1   by pixel content, reports/dup_audit.json
      v
distinct photo (cluster) --1:1--> cleaned photo, row in manifest.csv
                                    |-- 1:1 --> COCO image in exactly one split --1:N--> COCO annotations
                                    |-- 1:1 --> YOLO image and label file
                                    '-- N:1 --> scene --N:1--> stratum (kind x density_bucket)
                                                  '-- N:1 --> split  (a scene never spans two splits)
```

- `kind` is one of `indoor`, `outdoor`, `semi` or `unknown`. The fifth kind,
  `bukan-cctv`, is dropped before anything is written.
- `density_bucket` in the manifest is the **scene's** median crowding (`d0-2`,
  `d3-5`, `d6-10`, `d10+`). `evaluate.py` buckets frames by their **own** count
  instead (`0-2`, `3-5`, `6-10`, `10+`). The two are different measures on purpose.
- Where the base images came from (`split_in_export`) is kept for audit only. It
  plays no part in the new split.

## Store data (Stage 6)

```
data/store/<name>/images/<name>_<frame:07d>.jpg   frames from the operator's video
                /labels/<same stem>.txt           drafts from the model, later corrected by a human
                /classes.txt                      "person"
                /manifest.csv
        |
        v  merge_store_data.py
data/merged/images/{split}/, labels/{split}/      the base dataset copied in, plus the store frames
data/merged/data.yaml                             train on everything
data/merged/data_store_test.yaml                  val and test = test_store.txt (store frames only)
```

Store frames are split by **contiguous time block**: 10 blocks, with the first ones
going to test, the next to val and the rest to train. Two consequences follow from
the code:

- With the defaults the split is **test 20%, val 20%, train 60%**. The val share is
  not 15%, because `round(1.5)` is 2 in Python.
- Test is always the **start of the recording**. If the operator joins their four
  sessions in time order, as `PANDUAN.md` suggests, test is mostly the first session.

## Migration strategy

There are no migrations. To change the data, change the code and **regenerate**:

```
unzip source -> audit_duplicates.py -> clean_cctv.py -> make_figures.py -> train -> evaluate
```

`clean_cctv.py` alone is enough to change only the split, because
`reports/dup_audit.json` is committed; the raw images still have to be extracted first.

**Clear the outputs before regenerating.** `clean_cctv.py` and `merge_store_data.py`
create their directories with `exist_ok=True` and never delete anything. If the
split changes, a photo that moved from `val` to `train` stays in
`yolo/images/val/` as well, which is a train/test leak. Delete `data/cctv-person/`
(except the committed JSON, CSV and YAML you mean to replace) or `data/merged/`
before re-running.

## Data rules

1. Never group or deduplicate by filename. 1507 stems hold more than one distinct
   photo. Use pixel content (`audit_duplicates.py`).
2. Never split frames from one camera at random: base data by scene, store data by
   time block.
3. Join predictions to ground truth by the **input** file name, never by
   `basename(r.path)`. Ultralytics renames list inputs to `image0.jpg`,
   `image1.jpg`, and so on.
4. COCO image IDs are per split. Always look up an image by `(split, file_name)`.
5. Only categories 0-3 of the raw export are people; 4 and 5 are shoes. Check all
   three raw splits, because the shoes appear only in `valid`.
6. `data.yaml` holds an absolute path. Update it if the project moves.
7. The filesystem is case-insensitive. `clean_cctv.py` adds a `__2` suffix to names
   that differ only in case. After any bulk copy, compare the file count with the
   manifest.
8. Never train or evaluate on uncorrected draft labels from `data/store/`.
9. Camera frames contain identifiable people. Do not commit images, and do not
   upload footage to third parties without the user's decision.
