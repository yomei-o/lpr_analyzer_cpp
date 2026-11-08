"""Real frames with checked plate boxes (train/sample_plates.tsv) -> YOLO-format dirs for plate_det_torch.py.

  python train/make_real_det.py --tsv train/sample_plates.tsv --root .. --out data/det_real --val-out data/det_real_val
Every `--val-every`-th image goes to the validation dir instead (real-scene recall, not synthetic).
"""
import argparse
import collections
import os

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--tsv", required=True)
ap.add_argument("--root", default="..", help="directory that contains alpr_jp/")
ap.add_argument("--out", required=True)
ap.add_argument("--val-out", required=True)
ap.add_argument("--val-every", type=int, default=8)
a = ap.parse_args()
boxes = collections.defaultdict(list)
for line in open(a.tsv, encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    p = line.rstrip("\n").split("\t")
    boxes[p[0]].append([float(v) for v in p[1:5]])
for d in (a.out, a.val_out):
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
n = [0, 0]
for k, (path, bs) in enumerate(sorted(boxes.items())):
    img = cv2.imread(os.path.join(a.root, path))
    H, W = img.shape[:2]
    d = a.val_out if k % a.val_every == a.val_every - 1 else a.out
    n[d == a.val_out] += 1
    name = "real%04d" % k
    cv2.imwrite(os.path.join(d, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(d, "labels", name + ".txt"), "w") as o:
        for x1, y1, x2, y2 in bs:
            o.write("0 %.6f %.6f %.6f %.6f\n" % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
print("train", n[0], "val", n[1])
