"""認識器（ONNX）を合成の切り抜きセット（jp_crops_to_synth.py の labels.txt + corners.txt）で採点する。

  python train/eval_synth.py <model.onnx> <crops dir> [--yolo-lpr ../yolo_lpr_cpp] [--limit N]

head ごとの正解率と、版全体（地名・分類番号・ひらがな・4 桁がすべて正解）の割合を出す。
地名は、学習用の alpr_jp に出てくる地名とそれ以外にも分けて出す（見たことの無い地名への強さを見るため）。
切り出しは学習器の評価と同じく、4 隅の外接矩形＋余白 0.03。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def take(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    model, crops = sys.argv[1], sys.argv[2]
    yolo = os.path.abspath(take("--yolo-lpr", os.path.join(HERE, "..", "..", "yolo_lpr_cpp")))
    limit = int(take("--limit", "0"))
    sys.path.insert(0, os.path.join(yolo, "tools"))
    import infer as I
    import labels as L
    import onnxruntime as ort
    sp = L.load(os.path.join(yolo, "spec", "labels.txt"))
    order = [g.name for g in sp.of_kind("head")]
    alpr = os.path.join(yolo, "..", "alpr_jp")
    seen = set()
    for cat in ("自家用", "自家用(軽)", "事業用", "事業用(軽)"):
        d = os.path.join(alpr, cat)
        if os.path.isdir(d):
            seen |= {n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))}
    reg_tok = sp.head("region").tok
    boxes = {}
    for line in open(os.path.join(crops, "corners.txt"), encoding="utf-8"):
        q = line.split()
        v = [float(x) for x in q[1:9]]
        boxes[q[0]] = (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))
    items = []
    for line in open(os.path.join(crops, "labels.txt"), encoding="utf-8"):
        p = line.split()
        items.append((p[0], [int(x) for x in p[1:12]]))
    if limit:
        items = items[:limit]
    s = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    outs = [o.name for o in s.get_outputs()]
    heads = [h for h in order if h in outs]
    ok = {h: 0 for h in heads}
    full = 0
    rs = {True: [0, 0], False: [0, 0]}
    B = 64
    for b in range(0, len(items), B):
        chunk = items[b:b + B]
        x = np.concatenate([I.crop_to_input(I.load_rgb(os.path.join(crops, k)), *boxes[k], 0.03)
                            for k, _ in chunk]).astype(np.float32)
        res = dict(zip(outs, s.run(outs, {inp: x})))
        for j, (k, lab) in enumerate(chunk):
            allok = True
            for h in heads:
                if h in ("plate_kind", "legible"):
                    good = int(res[h][j].argmax()) == lab[order.index(h)]
                    ok[h] += good
                    continue
                good = int(res[h][j].argmax()) == lab[order.index(h)]
                ok[h] += good
                allok &= good
            full += allok
            r = rs[reg_tok[lab[0]] in seen]
            r[0] += int(res["region"][j].argmax()) == lab[0]
            r[1] += 1
    n = len(items)
    print("%s on %s (%d crops)" % (os.path.basename(model), crops, n))
    for h in heads:
        print("  %-13s %6.1f%%" % (h, 100.0 * ok[h] / n))
    print("  region, in alpr_jp     %6.1f%% (%d)" % (100.0 * rs[True][0] / max(1, rs[True][1]), rs[True][1]))
    print("  region, not in alpr_jp %6.1f%% (%d)" % (100.0 * rs[False][0] / max(1, rs[False][1]), rs[False][1]))
    print("  whole plate            %6.1f%%" % (100.0 * full / n))


if __name__ == "__main__":
    main()
