"""Does the ONNX that `jlpr init-det --from-pt best.pt` wrote compute what the PyTorch model computes?

  python train/check_export.py --pt runs/scratch/best.pt --onnx plate_det_scratch_640.onnx --imgsz 640

Runs both on the same random input and compares the final [1, 4+nc, anchors] output (cxcywh pixels,
sigmoid scores) -- the tensor the analyzer decodes.
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
from plate_det_torch import YOLOv8, make_anchors  # noqa: E402


def torch_output(model, x, reg=16):
    feats = model(x)
    anc, st = make_anchors(feats, YOLOv8.STRIDES)
    B = x.shape[0]
    dist = torch.cat([b.flatten(2) for b, _ in feats], 2).permute(0, 2, 1)
    cls = torch.cat([c.flatten(2) for _, c in feats], 2).permute(0, 2, 1).sigmoid()
    d = dist.view(B, -1, 4, reg).softmax(3) @ torch.arange(reg, dtype=torch.float32)
    x1y1, x2y2 = anc - d[..., :2], anc + d[..., 2:]
    box = torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], -1) * st
    return torch.cat([box, cls], -1).permute(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    m = YOLOv8(nc=1)
    missing, unexpected = m.load_state_dict(torch.load(a.pt, map_location="cpu"), strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)
    m.eval()
    x = torch.rand(1, 3, a.imgsz, a.imgsz)
    with torch.no_grad():
        ref = torch_output(m, x).numpy()
    s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    out = s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]
    print("torch", ref.shape, "onnx", out.shape)
    d = np.abs(out - ref)
    print(f"max |diff| boxes {d[:, :4].max():.3e}  scores {d[:, 4:].max():.3e}")
    ok = d[:, :4].max() < 1e-2 and d[:, 4:].max() < 1e-4
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
