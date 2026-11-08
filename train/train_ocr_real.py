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


FLAT = take("--flat-labels", os.path.join(HERE, "alpr_labels_flat.tsv"))
USE_FLAT = take("--use-flat", "1") == "1"


def load_flat(spec, root):
    """alpr_jp's 自家用(軽)/事業用/事業用(軽) crops sit in flat folders (no region folder), so the
    original loader skips all 553 of them -- including every dark plate (事業用: white on green,
    事業用(軽): yellow on black). train/alpr_labels_flat.tsv has the hand transcription of each."""
    heads = spec.of_kind("head")
    order = [g.name for g in heads]
    reg = spec.head("region")
    items, table = [], {}
    for line in open(FLAT, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        key, region, cls, hira, num = line.rstrip("\n").split("\t")
        path = os.path.join(root, key)
        if not os.path.exists(path):
            continue
        ridx = reg.tok.index(region) if region != "?" else -1
        items.append((path, max(ridx, 0), key))
        lab = {}
        fake = "\t".join([key, cls, hira, num, ""])
        tmp = load_table_line(spec, fake)
        lab.update(tmp)
        if ridx < 0:
            lab[order.index("region")] = None      # unknown region: masked
        table[key] = lab
    return items, table


def load_table_line(spec, line):
    import tempfile
    global LABELS
    keep = LABELS
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
        f.write(line + "\n")
        name = f.name
    LABELS = name
    try:
        t = load_table(spec)
    finally:
        LABELS = keep
        os.unlink(name)
    return next(iter(t.values()))


def _init(self, root, spec, hold_out=0.2):
    _orig_init(self, root, spec, hold_out)
    self.full = load_table(spec)
    hit = sum(1 for it in self.items if it[2] in self.full)
    print("alpr_labels: %d of %d real crops have hand labels" % (hit, len(self.items)), flush=True)
    if USE_FLAT and os.path.exists(FLAT):
        fitems, ftable = load_flat(spec, root)
        n0 = len(self.items)
        self.items = self.items + fitems
        self.full.update(ftable)
        # same 80/20 split rule for the new crops, deterministic, appended after the original split
        rng = np.random.default_rng(11)
        idx = list(range(n0, len(self.items)))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - hold_out))
        self.train_idx = sorted(self.train_idx + idx[:cut])
        self.val_idx = sorted(self.val_idx + idx[cut:])
        print("alpr flat folders: +%d crops (kei / commercial), %d train / %d hold-out in total"
              % (len(fitems), len(self.train_idx), len(self.val_idx)), flush=True)
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
        if v is None:               # region unknown for this crop: do not train that head on it
            mask[h] = 0
            continue
        heads[h] = v
        mask[h] = 1
    if isinstance(aug, tuple) and LOWRES_P > 0 and self.lowres_rng.random() < LOWRES_P:
        x = _lowres(x, self.lowres_rng)
    return x, heads, mask


T.AlprSet.__init__ = _init
T.AlprSet.load = _load


# ---- negatives: things that are not a plate -------------------------------------------------------
# A recognizer trained only on plates answers "a plate" with high confidence for anything it is shown
# (signs, bumpers, ...), so the analyzer cannot tell a misdetection from a plate. The `legible` head
# (spec: no / yes) becomes that signal: negatives teach legible = no and nothing else (every other head
# is masked), real plates teach legible = yes. Negatives are cut once at start-up and kept in memory.
NEG_TSV = os.path.abspath(take("--neg", "")) if "--neg" in sys.argv else ""
NEG_RANDOM = int(take("--neg-random", "0"))      # extra random 2:1 patches from these images (glob)
NEG_RANDOM_GLOB = take("--neg-random-glob", "")
NEG_P = float(take("--neg-p", "0.15"))           # share of each batch replaced by negatives
# Dark plates (事業用: white on green, 軽事業用: yellow on black) are 130 of the 720 real crops, and the
# from-scratch recognizer read them badly (IMG_7121's taxi 96-99 came out as 6698 / 8669). Two levers:
COMM_P = float(take("--commercial-p", "0"))      # share of real draws taken from the 事業用 folders
INVERT_P = float(take("--invert-p", "0"))        # real crops shown colour-inverted (light-on-dark text)
_PICK_PATCHED = False
LEGIBLE = None


def _cut(rgb, box, pad=0.25):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - bw * pad)), int(max(0, y1 - bh * pad))
    X2, Y2 = int(min(W, x2 + bw * pad)), int(min(H, y2 + bh * pad))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return None
    return rgb[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)


def load_negatives():
    import glob
    import infer as I
    negs = []
    if NEG_TSV:
        by_img = {}
        for line in open(NEG_TSV, encoding="utf-8"):
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                by_img.setdefault(p[0], []).append(tuple(float(v) for v in p[1:5]))
        for path, boxes in by_img.items():
            if not os.path.exists(path):                 # "alpr_jp/..." is relative to yolo_lpr_cpp's parent
                path = os.path.join(YOLO, "..", path)
            rgb = I.load_rgb(path)
            for b in boxes:
                c = _cut(rgb, b)
                if c:
                    negs.append(c)
    if NEG_RANDOM and NEG_RANDOM_GLOB:
        rng = np.random.default_rng(5)
        files = sorted(glob.glob(NEG_RANDOM_GLOB))
        for k in range(NEG_RANDOM):
            rgb = I.load_rgb(files[int(rng.integers(len(files)))])
            H, W = rgb.shape[:2]
            w = float(rng.uniform(40, min(400, W * 0.5)))
            h = w * float(rng.uniform(0.4, 0.7))
            x1, y1 = float(rng.uniform(0, W - w)), float(rng.uniform(0, H - h))
            c = _cut(rgb, (x1, y1, x1 + w, y1 + h))
            if c:
                negs.append(c)
    print("negatives: %d crops" % len(negs), flush=True)
    return negs


_orig_batch_from = T.batch_from
NEGS = None


def _batch_from(sets, weights, batch, rng, spec, pool=None):
    global NEGS, LEGIBLE, _PICK_PATCHED
    if not _PICK_PATCHED and COMM_P > 0:
        _PICK_PATCHED = True
        for st in sets:
            if isinstance(st, T.AlprSet):
                comm = [i for i in st.train_idx if st.items[i][2].startswith("事業用")]  # 事業用/ and 事業用(軽)/
                base = st.pick
                st.pick = (lambda r, c=comm, b=base: c[r.below(len(c))] if r.unit() < COMM_P else b(r))
                print("commercial oversampling: %d crops at p=%.2f" % (len(comm), COMM_P), flush=True)
    x, y, m = _orig_batch_from(sets, weights, batch, rng, spec, pool)
    if LEGIBLE is None:
        LEGIBLE = [g.name for g in spec.of_kind("head")].index("legible")
    if NEGS is None:
        NEGS = load_negatives() if (NEG_TSV or NEG_RANDOM) else []
    if not NEGS or NEG_P <= 0:
        return x, y, m
    import infer as I
    g = np.random.default_rng()
    k = int(g.binomial(batch, NEG_P))
    for j in range(k):
        img, (x1, y1, x2, y2) = NEGS[int(g.integers(len(NEGS)))]
        margin = float(g.uniform(-0.04, 0.12))
        v = I.crop_to_input(img, x1, y1, x2, y2, margin)[0]
        x[j] = __import__("torch").from_numpy(v).float()
        y[j] = 0
        y[j, LEGIBLE] = 0           # "no"
        m[j] = 0
        m[j, LEGIBLE] = 1
    return x, y, m


T.batch_from = _batch_from

_load_with_labels = T.AlprSet.load


def _load_legible(self, i, aug):
    x, heads, mask = _load_with_labels(self, i, aug)
    li = len(heads) - 1                 # legible is the last head in spec order
    heads[li] = 1                       # "yes": a real plate a person could read
    mask[li] = 1
    if INVERT_P > 0 and isinstance(aug, tuple) and np.random.random() < INVERT_P:
        x = (1.0 - x).astype(np.float32)
    return x, heads, mask


T.AlprSet.load = _load_legible


def eval_onnx(path):
    """Per-head accuracy of an exported recognizer on the real hold-out (hand labels as truth).
    Reported separately for the region-folder crops (the original 144) and the flat-folder ones
    (kei / commercial), so numbers from before the flat folders were added stay comparable."""
    import onnxruntime as ort
    spec = L.load(os.path.join(YOLO, "spec", "labels.txt"))
    ds = T.AlprSet(os.path.join(YOLO, "..", "alpr_jp"), spec)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    order = [g.name for g in spec.of_kind("head")]
    nums = [order.index(h) for h in ("plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04")]
    flat_prefix = ("自家用(軽)/", "事業用/", "事業用(軽)/")
    for title, pick in (("region folders", lambda k: not k.startswith(flat_prefix)),
                        ("kei / commercial (flat folders)", lambda k: k.startswith(flat_prefix))):
        ok = {n: [0, 0] for n in order}
        num_all = [0, 0]
        for i in ds.val_idx:
            key = ds.items[i][2]
            if not pick(key):
                continue
            x, heads, mask = _orig_load(ds, i, 0.0)
            lab = dict(ds.full.get(key, {}))
            outs = s.run(None, {s.get_inputs()[0].name: x[None].astype(np.float32)})
            pred = {n: int(np.argmax(o)) for n, o in zip(names, outs)}
            truth = {} if lab.get(0, 0) is None else {0: heads[0]}
            truth.update({h: v for h, v in lab.items() if v is not None})
            for h, v in truth.items():
                n = order[h]
                p = pred.get(n, pred.get(n + "_output"))
                if p is None:
                    continue
                ok[n][0] += int(p == v)
                ok[n][1] += 1
            if all(h in truth for h in nums):
                num_all[0] += int(all(pred.get(order[h], pred.get(order[h] + "_output")) == truth[h] for h in nums))
                num_all[1] += 1
        print("== " + title)
        for n in order:
            if ok[n][1]:
                print("  %-14s %5.1f%%  (%d)" % (n, 100 * ok[n][0] / ok[n][1], ok[n][1]))
        print("  4-digit serial all correct: %.1f%% (%d)" % (100 * num_all[0] / max(1, num_all[1]), num_all[1]))


if __name__ == "__main__":
    if EVAL_ONNX:
        eval_onnx(EVAL_ONNX)
    else:
        T.main()
