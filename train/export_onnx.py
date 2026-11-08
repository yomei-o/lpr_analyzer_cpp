"""Put trained weights into the graph yolo_lpr_cpp writes, giving the ONNX the analyzer runs.

  jlpr init-det --out base_640.onnx --arch n --nc 1 --imgsz 640          # random weights, C++ graph
  python train/export_onnx.py --pt runs/scratch/best.pt --graph base_640.onnx --out plate_det_scratch_640.onnx

The graph's initializers carry the same names as the PyTorch state_dict (`model.0.conv.weight`,
`model.0.bn.running_mean`, ...), so every tensor is replaced by name and shape. Anything left over on
either side is reported and fails the export.
"""
import argparse

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = {k: v.numpy() for k, v in torch.load(a.pt, map_location="cpu").items()}
    g = onnx.load(a.graph)
    used = set()
    replaced = 0
    weight_like = []
    for i, t in enumerate(g.graph.initializer):
        if t.name in sd:
            old = numpy_helper.to_array(t)
            new = sd[t.name].astype(np.float32)
            assert old.shape == new.shape, (t.name, old.shape, new.shape)
            g.graph.initializer[i].CopyFrom(numpy_helper.from_array(new, t.name))
            used.add(t.name)
            replaced += 1
        elif t.name.startswith("model.") and "/" not in t.name:   # "model.2/splitsz_14" are graph constants
            weight_like.append(t.name)
    left = sorted(set(sd) - used - {"model.22.dfl.conv.weight"})   # the graph has its own fixed 0..15 ramp
    print(f"replaced {replaced} tensors; graph-only model.* tensors: {weight_like}; state_dict-only: {left}")
    if weight_like or left:
        raise SystemExit("export: tensor sets differ")
    onnx.save(g, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
