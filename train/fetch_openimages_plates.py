"""Open Images V7 "Vehicle registration plate" (/m/01jfm_) -> YOLO-format detection data.

Real photos of plates of many countries, in every kind of scene -- the variety the synthetic
composites and one fixed gate camera do not have. License: annotations CC BY 4.0, images CC BY 2.0
(each image's author is in the Open Images metadata); a model trained on them is not a derivative
covered by those licenses, but credit Open Images (README / models_noagpl/README.md).

  python train/fetch_openimages_plates.py --out data/det_oi --split train --max 8000
  python train/fetch_openimages_plates.py --out data/det_oi_val --split validation

Images are downscaled to at most --max-side px (the trainer resizes to 640 anyway). Images with a
group box (IsGroupOf) or a depiction (a plate in a picture) are skipped: their boxes do not mark plates.
"""
import argparse
import collections
import csv
import io
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ANN = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
PLATE = "/m/01jfm_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=list(ANN))
    ap.add_argument("--max", type=int, default=0, help="at most this many images (0 = all)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cache", default="oi_cache")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    rows_path = os.path.join(a.cache, "%s_plates.csv" % a.split)
    if not os.path.exists(rows_path):
        # stream the (2.2 GB for train) annotation file through grep: only the plate rows are kept
        cmd = "curl -sS %s | grep '%s' > %s" % (ANN[a.split], PLATE, rows_path)
        subprocess.check_call(cmd, shell=True)
    boxes = collections.defaultdict(list)
    bad = set()
    for r in csv.reader(open(rows_path)):
        img, lab = r[0], r[2]
        if lab != PLATE:
            continue
        xmin, xmax, ymin, ymax = map(float, r[4:8])
        group, depiction = r[10] == "1", r[11] == "1"
        if group or depiction:
            bad.add(img)
        boxes[img].append((xmin, ymin, xmax, ymax))
    ids = sorted(k for k in boxes if k not in bad)
    if a.max:
        rng = np.random.default_rng(0)
        ids = sorted(rng.choice(ids, size=min(a.max, len(ids)), replace=False).tolist())
    print("%s: %d images with plates (%d skipped as group/depiction)" % (a.split, len(ids), len(bad)), flush=True)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)

    def one(i):
        dst = os.path.join(a.out, "images", i + ".jpg")
        if os.path.exists(dst):
            return 1
        try:
            data = urllib.request.urlopen("https://s3.amazonaws.com/open-images-dataset/%s/%s.jpg" % (a.split, i),
                                          timeout=30).read()
        except Exception:
            return 0
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 0
        h, w = img.shape[:2]
        s = a.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(os.path.join(a.out, "labels", i + ".txt"), "w") as o:
            for x1, y1, x2, y2 in boxes[i]:
                o.write("0 %.6f %.6f %.6f %.6f\n" % ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        return 1

    with ThreadPoolExecutor(a.workers) as ex:
        got = 0
        for k, ok in enumerate(ex.map(one, ids)):
            got += ok
            if k % 500 == 0:
                print(k, got, flush=True)
    print("done: %d images in %s" % (got, a.out), flush=True)


if __name__ == "__main__":
    main()
