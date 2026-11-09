"""認識器（ONNX）を、解析器が実際に渡した入力（lpr_cli --dump）でコマ単位に採点する。

  python train/eval_frames.py train/video_frames model.onnx [model2.onnx ...]

<dump dir>/labels.tsv（file region cls hira num）は、dump の index.tsv の追跡 ID を
score_video.py の正解に引き当てて作ったもの。入力はもう 128x128 なので、そのまま /255 で流す。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "yolo_lpr_cpp", "tools"))


def main():
    import labels as L
    import onnxruntime as ort
    from PIL import Image
    d = sys.argv[1]
    sp = L.load(os.path.join(HERE, "..", "..", "yolo_lpr_cpp", "spec", "labels.txt"))
    tok = {g.name: g.tok for g in sp.of_kind("head")}
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "labels.tsv"), encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    xs = [np.asarray(Image.open(os.path.join(d, r[0])).convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
          for r in rows]
    for m in sys.argv[2:]:
        s = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0].name
        outs = [o.name for o in s.get_outputs()]
        ok = {"region": 0, "cls": 0, "hira": 0, "num": 0}
        per = {}
        for r, x in zip(rows, xs):
            res = dict(zip(outs, s.run(outs, {inp: x})))
            g = {h: tok[h][int(res[h][0].argmax())] for h in outs if h in tok}
            cls = "".join(g["class_num_%02d" % i] for i in (1, 2, 3)).replace("<blank>", "")
            num = "".join("." if g["plate_num_%02d" % i] == "<blank>" else g["plate_num_%02d" % i] for i in (1, 2, 3, 4))
            hit = {"region": g["region"] == r[1], "cls": cls == r[2], "hira": g["hiragana"] == r[3], "num": num == r[4]}
            for k in ok:
                ok[k] += hit[k]
            p = per.setdefault(r[1] + " " + r[4], [0, 0, {}])
            p[0] += hit["region"]
            p[1] += 1
            p[2][g["region"]] = p[2].get(g["region"], 0) + 1
        n = len(rows)
        print("%-28s n=%d  地名 %5.1f%%  分類番号 %5.1f%%  ひらがな %5.1f%%  4桁 %5.1f%%" % (
            os.path.basename(m), n, *(100.0 * ok[k] / n for k in ("region", "cls", "hira", "num"))))
        print("    " + "  ".join("%s %d/%d(%s)" % (k, v[0], v[1], max(v[2], key=v[2].get)) for k, v in per.items()))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
