"""Show where every training run is right now.

Ultralytics resolves a RELATIVE `project=` against its own global `runs_dir`
setting, not the working directory. On this machine that setting points at an
unrelated project, so output landed outside this repo. This script finds the
real directory either way, so progress can be checked without hunting for it.

    python tools/status.py
    python tools/status.py --watch      # refresh every 30s
"""
import argparse
import csv
import json
import os
import time
from datetime import timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER = ["default", "tuned", "indoor"]
EPOCHS_PLANNED = 150


def runs_root():
    """Where ultralytics actually writes, honouring its global runs_dir."""
    local = os.path.join(ROOT, "runs", "person")
    candidates = [local]
    try:
        cfg = json.load(open(os.path.expandvars(
            r"%APPDATA%\Ultralytics\settings.json")))
        rd = cfg.get("runs_dir")
        if rd:
            candidates.insert(0, os.path.join(rd, "detect", "runs", "person"))
            candidates.insert(1, os.path.join(rd, "runs", "person"))
    except Exception:
        pass
    for c in candidates:
        if os.path.isdir(c) and any(
                os.path.exists(os.path.join(c, n, "results.csv")) for n in ORDER):
            return c
    return local


def read_rows(p):
    with open(p) as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()

    while True:
        root = runs_root()
        print(f"\n{'=' * 78}")
        print(f"output ultralytics: {root}")
        print(f"waktu             : {time.strftime('%H:%M:%S')}")
        print("=" * 78)
        print(f"{'run':<9}{'epoch':>10}{'mAP50':>9}{'mAP50-95':>10}"
              f"{'P':>8}{'R':>8}{'berjalan':>11}{'status':>12}")
        print("-" * 78)

        for n in ORDER:
            d = os.path.join(root, n)
            csv_p = os.path.join(d, "results.csv")
            best = os.path.join(d, "weights", "best.pt")
            if not os.path.exists(csv_p):
                state = "selesai" if os.path.exists(best) else (
                    "memulai" if os.path.isdir(d) else "antri")
                print(f"{n:<9}{'-':>10}{'-':>9}{'-':>10}{'-':>8}{'-':>8}"
                      f"{'-':>11}{state:>12}")
                continue
            rows = read_rows(csv_p)
            if not rows:
                print(f"{n:<9}{'0':>10}{'-':>9}{'-':>10}{'-':>8}{'-':>8}"
                      f"{'-':>11}{'memulai':>12}")
                continue
            r = rows[-1]
            ep = int(float(r["epoch"]))
            el = timedelta(seconds=int(float(r.get("time", 0))))
            # train.py writes run_meta.json only after model.train() returns, so
            # it is the reliable "finished" marker -- epoch count is not, because
            # patience=30 stops well short of EPOCHS_PLANNED (default ended at 131)
            done = os.path.exists(os.path.join(ROOT, "runs", "person", n,
                                               "run_meta.json"))
            idle = time.time() - os.path.getmtime(csv_p)
            state = "SELESAI" if done else ("jalan" if idle < 600 else "berhenti?")
            print(f"{n:<9}{f'{ep}/{EPOCHS_PLANNED}':>10}"
                  f"{float(r['metrics/mAP50(B)']):>9.4f}"
                  f"{float(r['metrics/mAP50-95(B)']):>10.4f}"
                  f"{float(r['metrics/precision(B)']):>8.4f}"
                  f"{float(r['metrics/recall(B)']):>8.4f}"
                  f"{str(el):>11}{state:>12}")

        print("\nangka di atas dari split VAL, dihitung ultralytics tiap epoch.")
        print("evaluasi resmi di split TEST: python tools/evaluate.py --all")
        if not a.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
