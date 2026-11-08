"""Recognizer training from random init, with the real plates labelled on every head.

yolo_lpr_cpp's tools/train_ocr.py knows only the region of each alpr_jp crop (the folder name), so on
real images every head except the region is masked out, and the digits are learned from synthetic
glyphs alone. That is fine when you start from weights that have already seen real plates; from random
initialisation it is not: measured on IMG_7121, a recognizer trained that way read 12-09 as ・531.

train/alpr_labels.tsv is a hand transcription of the same 720 crops (class number, hiragana, serial),
so this wrapper un-masks those heads for the real crops, fixes the two folders that hold another
region's plate, and adds a resolution augmentation (video plates are 60-120 px wide, the crops are
mostly 200-400). Everything else is train_ocr.py unchanged, imported from the yolo_lpr_cpp checkout:

  python train/train_ocr_real.py --yolo-lpr ../yolo_lpr_cpp <train_ocr.py arguments>
  python train/train_ocr_real.py --yolo-lpr ../yolo_lpr_cpp --eval-onnx model.onnx   # per-head hold-out accuracy
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def take(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


YOLO = os.path.abspath(take("--yolo-lpr", os.path.join(HERE, "..", "..", "yolo_lpr_cpp")))
LABELS = take("--labels", os.path.join(HERE, "alpr_labels.tsv"))
EVAL_ONNX = take("--eval-onnx", "")
LOWRES_P = float(take("--lowres-p", "0.5"))
sys.path.insert(0, os.path.join(YOLO, "tools"))
os.chdir(YOLO)                      # train_ocr.py resolves models/ and spec/ relative to its repo
import train_ocr as T  # noqa: E402
import labels as L  # noqa: E402


def load_table(spec):
    heads = spec.of_kind("head")
    tok = {g.name: g.tok for g in heads}
    order = [g.name for g in heads]
    reg = spec.head("region")
    table = {}
    for line in open(LABELS, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        key, cls, hira, num, region = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
        lab = {}
        if region:
            lab["region"] = reg.tok.index(region)
        if cls != "?":
            c = list(cls) + ["<blank>"] * (3 - len(cls))
            for h, ch in zip(("class_num_01", "class_num_02", "class_num_03"), c):
                if ch in tok[h]:
                    lab[h] = tok[h].index(ch)
        if hira != "?" and hira in tok["hiragana"]:
            lab["hiragana"] = tok["hiragana"].index(hira)
        if num != "?" and len(num) == 4:
            for h, ch in zip(("plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04"), num):
                t = "<blank>" if ch == "." else ch
                if t in tok[h]:
                    lab[h] = tok[h].index(t)
        table[key] = {order.index(h): v for h, v in lab.items()}
    return table


_orig_init = T.AlprSet.__init__
_orig_load = T.AlprSet.load


def _init(self, root, spec, hold_out=0.2):
    _orig_init(self, root, spec, hold_out)
    self.full = load_table(spec)
    hit = sum(1 for it in self.items if it[2] in self.full)
    print("alpr_labels: %d of %d real crops have hand labels" % (hit, len(self.items)), flush=True)
    self.lowres_rng = np.random.default_rng(7)


def _lowres(x, rng):
    # downscale the 128x128 input to a video-like resolution and back (nearest-ish blur)
    import torch
    import torch.nn.functional as F
    s = int(rng.integers(28, 72))
    t = torch.from_numpy(x[None])
    t = F.interpolate(t, size=(s, s), mode="area")
    t = F.interpolate(t, size=(x.shape[1], x.shape[2]), mode="bilinear", align_corners=False)
    return t[0].numpy().astype(np.float32)


def _load(self, i, aug):
    x, heads, mask = _orig_load(self, i, aug)
    lab = self.full.get(self.items[i][2], {})
    for h, v in lab.items():
        heads[h] = v
        mask[h] = 1
    if isinstance(aug, tuple) and LOWRES_P > 0 and self.lowres_rng.random() < LOWRES_P:
        x = _lowres(x, self.lowres_rng)
    return x, heads, mask


T.AlprSet.__init__ = _init
T.AlprSet.load = _load


def eval_onnx(path):
    """Per-head accuracy of an exported recognizer on the real hold-out (hand labels as truth)."""
    import onnxruntime as ort
    spec = L.load(os.path.join(YOLO, "spec", "labels.txt"))
    ds = T.AlprSet(os.path.join(YOLO, "..", "alpr_jp"), spec)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    order = [g.name for g in spec.of_kind("head")]
    ok = {n: [0, 0] for n in order}
    num_all = [0, 0]
    for i in ds.val_idx:
        x, heads, mask = _orig_load(ds, i, 0.0)
        lab = ds.full.get(ds.items[i][2], {})
        outs = s.run(None, {s.get_inputs()[0].name: x[None].astype(np.float32)})
        pred = {n: int(np.argmax(o)) for n, o in zip(names, outs)}
        truth = {0: heads[0]}
        truth.update(lab)
        for h, v in truth.items():
            n = order[h]
            key = n if n in pred else n + "_output"
            p = pred.get(key, pred.get(n))
            if p is None:
                continue
            ok[n][0] += int(p == v)
            ok[n][1] += 1
        nums = [order.index(h) for h in ("plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04")]
        if all(h in lab for h in nums):
            num_all[0] += int(all(pred.get(order[h], pred.get(order[h] + "_output")) == lab[h] for h in nums))
            num_all[1] += 1
    for n in order:
        if ok[n][1]:
            print("  %-14s %5.1f%%  (%d)" % (n, 100 * ok[n][0] / ok[n][1], ok[n][1]))
    print("  4-digit serial all correct: %.1f%% (%d)" % (100 * num_all[0] / max(1, num_all[1]), num_all[1]))


if __name__ == "__main__":
    if EVAL_ONNX:
        eval_onnx(EVAL_ONNX)
    else:
        T.main()
