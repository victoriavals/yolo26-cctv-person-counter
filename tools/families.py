"""Scene families recovered from the Roboflow filename stems.

The export has no metadata saying where a photo came from, but the original
filenames survived the export and fall into a handful of naming conventions.
Each pattern below was checked against random contact sheets
(`reports/figures/sheet_cctv_*.png`) before its label was assigned.

`kind` is what the rest of the pipeline keys off:
  indoor      ceiling CCTV inside shops, warehouses, offices, corridors
  outdoor     street, parking, campus, park CCTV
  semi        entrances -- the indoor/outdoor threshold
  bukan-cctv  webcams and selfies scraped from YouTube; wrong angle, distance
              and lens for this project, so they are dropped before training
  unknown     filename has no usable pattern; mixed indoor and outdoor

Match against the ORIGINAL filename or its stem, not the derived scene name --
`scene_of()` strips trailing frame indices, which breaks patterns like
`^image[_-]?\\d`.
"""
import re

FAMILIES = [
    (r"^01-08-2022__",            "toko / gudang / kantor", "indoor"),
    (r"^image[_-]?\d",            "koridor / lobi",         "indoor"),
    (r"^opencv_frame",            "kantor / lab",           "indoor"),
    (r"^output_mp4",              "lobi / pintu masuk",     "indoor"),
    (r"^Day_",                    "jalan / parkiran",       "outdoor"),
    (r"^SeieeDept",               "kampus",                 "outdoor"),
    (r"^(CrossRoad|Road|Bridge)", "jalan raya",             "outdoor"),
    (r"^scene\d",                 "taman",                  "outdoor"),
    (r"^\d{6}",                   "pintu masuk",            "semi"),
    (r"^youtube",                 "webcam / selfie",        "bukan-cctv"),
]

DROP_KINDS = {"bukan-cctv"}


def family_of(name):
    """-> (label, kind). `name` is an original filename or its stem."""
    for pat, label, kind in FAMILIES:
        if re.match(pat, name):
            return label, kind
    return "belum teridentifikasi", "unknown"


def kind_of(name):
    return family_of(name)[1]
