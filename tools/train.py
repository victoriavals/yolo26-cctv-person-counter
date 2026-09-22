"""Train YOLO26s on the cleaned CCTV-person dataset.

Every experiment is one entry in RUNS so the whole comparison is reproducible
from a single file. Results land in runs/person/<name>/.

    python tools/train.py --list
    python tools/train.py smoke          # 2 epochs, 8% of data -- proves VRAM fits
    python tools/train.py tuned
    python tools/train.py default tuned indoor
    python tools/train.py all

Baseline (pretrained yolo26s, no training) is not here -- it needs no training
loop and lives in tools/evaluate.py.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import kind_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOLO_DIR = os.path.join(ROOT, "data", "cctv-person", "yolo").replace("\\", "/")
DATA = f"{YOLO_DIR}/data.yaml"
DATA_INDOOR = f"{YOLO_DIR}/data_indoor.yaml"
MODEL = os.path.join(ROOT, "models", "yolo26s.pt").replace("\\", "/")

# Shared across every run so differences can only come from the overrides below.
COMMON = dict(
    model=MODEL,
    data=DATA,
    imgsz=640,          # the source images are 640x640; going higher invents no detail
    epochs=150,
    patience=30,        # ultralytics default is 100, i.e. effectively no early stop
    batch=16,           # 8 GB VRAM; drop to 8 if the smoke run OOMs
    cache=True,         # 300 MB of images against 34 GB RAM -- large speedup, no risk
    amp=True,
    # Windows spawns dataloader workers as full processes. With the ultralytics
    # default of 8, training keeps a train loader AND a val loader alive at once
    # -- 16 worker processes plus their pin-memory threads -- and validation dies
    # with "Pin memory thread exited unexpectedly". Probed in isolation, val()
    # survives 0/2/4/8 workers equally, so the count alone is not the fault; it
    # is the two loaders together. 2 is plenty here: images are cached in RAM,
    # so workers only run augmentation. (Measured on val: workers=0 took 8s,
    # workers=8 took 17s -- on a dataset this small, spawn overhead dominates.)
    workers=2,
    seed=0,
    deterministic=True,
    # MUST be absolute. Ultralytics resolves a relative `project=` against its
    # own global runs_dir in %APPDATA%\Ultralytics\settings.json, which on this
    # machine points at an unrelated project -- the first three runs landed in
    # D:\computer-vision\density-aware-yolo26-vehicle-counting\runs\detect\runs\person
    # instead of here. tools/status.py knows how to find both locations.
    project=os.path.join(ROOT, "runs", "person").replace("\\", "/"),
    plots=True,
    val=True,
    exist_ok=True,
)

# Augmentation reasoning, measured from this dataset rather than assumed:
#
#   The photos are ALREADY augmented in-pixel by the Roboflow fork -- rotation
#   with corner fill, salt-and-pepper noise, exposure shifts. Layering the same
#   transforms again compounds the degradation.
#
#   The deployment camera is a FIXED ceiling CCTV. It does not rotate, does not
#   zoom, and its white balance does not drift. Teaching the model to survive
#   variation that will never occur spends capacity for nothing.
#
#   Mosaic is the deliberate exception and stays at 1.0. It composites 4 images
#   into one, multiplying people-per-sample -- the only mechanism here that
#   offsets this dataset's real ceiling, where 50% of frames hold <=2 people.
#   Lowering it would hurt exactly the crowded case a counter cares about.
TUNED_AUG = dict(
    hsv_v=0.15,         # default 0.4 -- exposure variance already baked into pixels
    hsv_s=0.4,          # default 0.7 -- CCTV white balance is fixed
    scale=0.3,          # default 0.5 -- fixed camera, bounded subject distance
    degrees=0.0,        # already the default; rotation is baked in and unrealistic here
    mosaic=1.0,         # kept -- see note above
    close_mosaic=20,    # default 10 -- more clean epochs at the end
    fliplr=0.5,         # kept -- a mirrored store layout is plausible
)

# Fine-tuning the already-fine-tuned model on store footage is a DIFFERENT job
# from the runs above: the weights are already close, the store set is small, and
# the goal is to add one missing shape (seated, occluded people) without washing
# out what the model already knows. Hence: start from best.pt not COCO, fewer
# epochs, and a 10x lower learning rate so the existing weights are nudged rather
# than overwritten.
STORE_FT = dict(
    model=os.path.join(ROOT, "runs", "person", "default", "weights",
                       "best.pt").replace("\\", "/"),
    data=os.path.join(ROOT, "data", "merged", "data.yaml").replace("\\", "/"),
    epochs=60,
    patience=15,
    lr0=0.001,          # default 0.01 -- too aggressive for a warm start
    warmup_epochs=1.0,  # default 3.0 -- the model is not cold
)

RUNS = {
    # 2 epochs on 8% of the data: proves the batch fits in 8 GB and the whole
    # pipeline runs, without spending an hour to find out it does not.
    "smoke": dict(name="smoke", epochs=2, fraction=0.08, patience=2, cache=False,
                  plots=False),
    "default": dict(name="default"),
    "tuned": dict(name="tuned", **TUNED_AUG),
    "indoor": dict(name="indoor", data=DATA_INDOOR, **TUNED_AUG),
    # requires tools/merge_store_data.py to have been run first
    "store": dict(name="store", **STORE_FT),
}


def write_indoor_yaml():
    """data_indoor.yaml: indoor-only TRAIN, but the same full val and test.

    Keeping the eval splits identical is what makes the indoor run comparable
    to the others -- only the training distribution changes.
    """
    man = list(csv.DictReader(open(os.path.join(ROOT, "data/cctv-person/manifest.csv"))))
    indoor = [r["file"] for r in man
              if r["split"] == "train" and r.get("kind", kind_of(r["file"])) == "indoor"]
    listing = f"{YOLO_DIR}/train_indoor.txt"
    with open(listing, "w") as f:
        f.write("\n".join(f"{YOLO_DIR}/images/train/{fn}" for fn in indoor) + "\n")
    with open(DATA_INDOOR, "w") as f:
        f.write(
            "# Indoor-only training set; val and test stay identical to data.yaml\n"
            "# so the indoor run is directly comparable to the others.\n"
            f"path: {YOLO_DIR}\n"
            "train: train_indoor.txt\n"
            "val: images/val\ntest: images/test\n\nnc: 1\nnames: [person]\n")
    print(f"data_indoor.yaml -> {len(indoor)} gambar train indoor")
    return len(indoor)


def run_one(key):
    from ultralytics import YOLO
    cfg = dict(COMMON)
    cfg.update(RUNS[key])
    model_path = cfg.pop("model")
    name = cfg["name"]

    print(f"\n{'=' * 64}\nRUN: {name}\n{'=' * 64}")
    for k in sorted(set(RUNS[key]) - {"name"}):
        print(f"  {k} = {RUNS[key][k]}")

    t0 = time.time()
    model = YOLO(model_path)
    model.train(**cfg)
    dt = time.time() - t0

    out = os.path.join(ROOT, "runs", "person", name)
    meta = {"run": name, "seconds": round(dt, 1), "overrides": RUNS[key],
            "model": model_path, "weights": f"{out}/weights/best.pt"}
    os.makedirs(out, exist_ok=True)
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
    print(f"\n{name} selesai dalam {dt / 60:.1f} menit -> {out}/weights/best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="nama run, atau 'all'")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not a.runs:
        print("run tersedia:", ", ".join(RUNS))
        print("\nkonfigurasi bersama:")
        for k, v in COMMON.items():
            print(f"  {k:<14}{v}")
        return

    # 'all' means the real experiments; the smoke run is opt-in only
    # 'all' means the three public-data experiments; smoke and store are opt-in
    names = ([n for n in RUNS if n not in ("smoke", "store")]
             if a.runs == ["all"] else a.runs)
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        sys.exit(f"run tidak dikenal: {unknown}. Tersedia: {list(RUNS)}")

    if "indoor" in names:
        write_indoor_yaml()
    if "store" in names and not os.path.exists(RUNS["store"]["data"]):
        sys.exit(f"{RUNS['store']['data']} belum ada. Jalankan dulu: "
                 "python tools/merge_store_data.py --store data/store/<nama>")
    os.chdir(ROOT)
    for n in names:
        run_one(n)


if __name__ == "__main__":
    main()
