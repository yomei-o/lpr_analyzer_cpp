"""Plate box candidates on real photos, from our own (non-AGPL) detector, for a person to verify.

The from-scratch detector is weak on real street scenes (trained on synthetic composites only). To fix
that it needs real frames with plate boxes, and the only clean way to get them is: propose with our own
model, then check every proposal by eye. To miss as little as possible the proposals come from the whole
frame AND from overlapping tiles (a 1920-wide frame squashed to 640 leaves a plate ~30 px wide; a
960-px tile gives it ~55 px), at a low threshold -- false proposals are rejected at the verification step.

  python train/propose_boxes.py --pt models_noagpl/plate_det_scratch_best.pt \
      --images "../alpr_jp/train/sample/*/*.jpg" --out proposals.tsv
  -> proposals.tsv: <image>\t<x1>\t<y1>\t<x2>\t<y2>\t<score>   (pixels of the original image)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from plate_det_torch import YOLOv8, decode  # noqa: E402


def run(model, img, dev, S=640):
    x = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1).copy()
    x = torch.from_numpy(x)[None].to(dev).float() / 255
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        feats = model(x)
    (b, s), = decode(feats, YOLOv8.STRIDES, 16, conf=0.05, iou=0.5)
    h, w = img.shape[:2]
    b = b.float().cpu()
    b[:, [0, 2]] *= w / S
    b[:, [1, 3]] *= h / S
    return b, s.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--tile", type=int, default=960)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YOLOv8(nc=1)
    m.load_state_dict(torch.load(a.pt, map_location="cpu"), strict=False)
    m.to(dev).eval()
    files = sorted(glob.glob(a.images))
    print(len(files), "images", flush=True)
    with open(a.out, "w", encoding="utf-8") as o:
        for k, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                continue
            H, W = img.shape[:2]
            bs, ss = [], []
            b, s = run(m, img, dev)
            bs.append(b)
            ss.append(s)
            T = min(a.tile, W, H)
            xs = sorted(set([0, max(0, W - T)] + list(range(0, max(1, W - T), T // 2))))
            ys = sorted(set([0, max(0, H - T)] + list(range(0, max(1, H - T), T // 2))))
            for y0 in ys:
                for x0 in xs:
                    b, s = run(m, img[y0:y0 + T, x0:x0 + T], dev)
                    b[:, [0, 2]] += x0
                    b[:, [1, 3]] += y0
                    bs.append(b)
                    ss.append(s)
            b, s = torch.cat(bs), torch.cat(ss)
            keep = torchvision.ops.nms(b, s, 0.4)
            for i in keep.tolist():
                if s[i] < a.conf:
                    continue
                x1, y1, x2, y2 = b[i].tolist()
                o.write("%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.3f\n" % (f, x1, y1, x2, y2, float(s[i])))
            if k % 100 == 0:
                print(k, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
