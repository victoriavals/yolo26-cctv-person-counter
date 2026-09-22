"""Merge your labelled store frames into the training set, and build a
store-only test set to measure on.

Two outputs, because they answer different questions:

  data/merged/          the CCTV-person dataset + your frames -> fine-tune here
  test_store.txt        ONLY your frames from the test split -> the number that
                        actually predicts field behaviour

That second file is the point. A test-split mAP50 of 0.891 on the public data
did not predict this model missing 60-80% of people on real warehouse footage.
Only frames from your own camera can tell you whether the next run fixed it.

Your frames are split by TIME BLOCK, not at random. Frames seconds apart look
almost identical, so a random split would leak them across train and test and
inflate the score -- the same defect that made the original Roboflow split
meaningless.

    python tools/merge_store_data.py --store data/store/toko
    python tools/merge_store_data.py --store data/store/toko --store-test 0.25
"""
import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT, "data", "cctv-person", "yolo")
OUT = os.path.join(ROOT, "data", "merged")
SPLITS = ("train", "val", "test")


def time_blocks(stems, n_blocks):
    """Contiguous blocks over frame order, so neighbours stay on one side."""
    stems = sorted(stems)
    if n_blocks <= 1 or len(stems) < n_blocks:
        return [stems]
    size = len(stems) // n_blocks
    return [stems[i * size:(i + 1) * size if i < n_blocks - 1 else len(stems)]
            for i in range(n_blocks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="folder hasil pelabelan")
    ap.add_argument("--store-val", type=float, default=0.15)
    ap.add_argument("--store-test", type=float, default=0.20,
                    help="porsi frame toko untuk test -- sengaja lebih besar "
                         "dari 10%% biasa, karena inilah angka yang menentukan")
    a = ap.parse_args()

    store = os.path.abspath(a.store)
    s_img, s_lab = os.path.join(store, "images"), os.path.join(store, "labels")
    if not os.path.isdir(s_img):
        sys.exit(f"tidak ada {s_img}")
    if not os.path.isdir(BASE):
        sys.exit(f"dataset dasar tidak ada: {BASE}")

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(s_img)
                   if f.lower().endswith((".jpg", ".png")))
    if not stems:
        sys.exit("folder gambar kosong")

    missing = [s for s in stems if not os.path.exists(f"{s_lab}/{s}.txt")]
    if missing:
        sys.exit(f"{len(missing)} gambar belum punya file label, contoh: {missing[:3]}\n"
                 "Export dari alat pelabelan dulu (format YOLO), baru jalankan ini.")

    empty = sum(1 for s in stems
                if os.path.getsize(f"{s_lab}/{s}.txt") == 0)
    boxes = sum(sum(1 for line in open(f"{s_lab}/{s}.txt") if line.strip())
                for s in stems)
    print(f"frame toko : {len(stems)}  ({empty} tanpa orang)")
    print(f"kotak      : {boxes}  ({boxes / len(stems):.2f} per frame)")
    if boxes == 0:
        sys.exit("tidak ada satu pun kotak -- labelnya belum diisi?")

    # 10 time blocks, then hand whole blocks to val/test
    blocks = time_blocks(stems, 10)
    n_test = max(1, round(len(blocks) * a.store_test))
    n_val = max(1, round(len(blocks) * a.store_val))
    assign = {}
    for i, blk in enumerate(blocks):
        where = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for s in blk:
            assign[s] = where

    for sp in SPLITS:
        os.makedirs(f"{OUT}/images/{sp}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{sp}", exist_ok=True)

    copied = {sp: 0 for sp in SPLITS}
    for sp in SPLITS:
        for f in os.listdir(f"{BASE}/images/{sp}"):
            shutil.copyfile(f"{BASE}/images/{sp}/{f}", f"{OUT}/images/{sp}/{f}")
            lab = os.path.splitext(f)[0] + ".txt"
            if os.path.exists(f"{BASE}/labels/{sp}/{lab}"):
                shutil.copyfile(f"{BASE}/labels/{sp}/{lab}", f"{OUT}/labels/{sp}/{lab}")
            copied[sp] += 1
    print(f"dataset dasar disalin: {copied}")

    store_rows, added = [], {sp: 0 for sp in SPLITS}
    for s in stems:
        sp = assign[s]
        src_img = next(f"{s_img}/{s}{e}" for e in (".jpg", ".png")
                       if os.path.exists(f"{s_img}/{s}{e}"))
        ext = os.path.splitext(src_img)[1]
        shutil.copyfile(src_img, f"{OUT}/images/{sp}/{s}{ext}")
        shutil.copyfile(f"{s_lab}/{s}.txt", f"{OUT}/labels/{sp}/{s}.txt")
        added[sp] += 1
        store_rows.append({"file": f"{s}{ext}", "split": sp})
    print(f"frame toko ditambahkan: {added}")

    root = OUT.replace("\\", "/")
    test_store = f"{OUT}/test_store.txt"
    with open(test_store, "w") as f:
        f.write("\n".join(f"{root}/images/test/{r['file']}"
                          for r in store_rows if r["split"] == "test") + "\n")

    with open(f"{OUT}/data.yaml", "w") as f:
        f.write("# CCTV-person + frame dari kamera toko sendiri\n"
                f"path: {root}\n"
                "train: images/train\nval: images/val\ntest: images/test\n\n"
                "nc: 1\nnames: [person]\n")
    with open(f"{OUT}/data_store_test.yaml", "w") as f:
        f.write("# val/test HANYA frame toko: angka inilah yang berlaku di lapangan\n"
                f"path: {root}\n"
                "train: images/train\nval: test_store.txt\ntest: test_store.txt\n\n"
                "nc: 1\nnames: [person]\n")
    with open(f"{OUT}/store_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "split"])
        wr.writeheader(); wr.writerows(store_rows)

    n_store_test = sum(1 for r in store_rows if r["split"] == "test")
    print(f"\n-> {OUT}")
    print("   data.yaml             gabungan, untuk fine-tune")
    print(f"   data_store_test.yaml  evaluasi HANYA di {n_store_test} frame toko")
    if n_store_test < 50:
        print(f"\nPERINGATAN: hanya {n_store_test} frame toko di test. Di bawah ~50 frame,")
        print("angka mAP-nya terlalu berisik untuk dipercaya. Label lebih banyak.")
    print("\nlangkah berikut:")
    print("  python tools/train.py --list   # salin blok COMMON, ganti data= ke data/merged/data.yaml")
    print("  # mulai dari bobot yang ada, bukan dari COCO lagi:")
    print("  #   model=runs/person/default/weights/best.pt")


if __name__ == "__main__":
    main()
