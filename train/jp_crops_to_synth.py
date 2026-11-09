"""jp_lpr_image_cpp の切り抜き（--crops）を yolo_lpr_cpp の学習器の合成セット形式にする。

  python train/jp_crops_to_synth.py <out>/crops [--spec ../yolo_lpr_cpp/spec/labels.txt]

<crops>/labels.tsv（key region cls hira num）と <crops>/corners.txt（key + 4 隅）から、
<crops>/labels.txt（key + 11 head の番号）を書く。corners.txt はそのまま SynthSet が読む形式。
plate_kind は種別フォルダから、legible は yes。spec に無い字があれば、その行は捨てる。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "yolo_lpr_cpp", "tools"))


def main():
    crops = sys.argv[1]
    spec_path = os.path.join(HERE, "..", "..", "yolo_lpr_cpp", "spec", "labels.txt")
    if "--spec" in sys.argv:
        spec_path = sys.argv[sys.argv.index("--spec") + 1]
        sys.path.insert(0, os.path.join(os.path.dirname(spec_path), "..", "tools"))
    import labels as L
    sp = L.load(spec_path)
    tok = {g.name: g.tok for g in sp.of_kind("head")}
    order = [g.name for g in sp.of_kind("head")]
    kind = {"自家用": "private_normal", "事業用": "commercial_normal",
            "自家用(軽)": "kei_private", "事業用(軽)": "kei_commercial"}
    out, bad = [], 0
    for line in open(os.path.join(crops, "labels.tsv"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        key, region, cls, hira, num = line.rstrip("\r\n").split("\t")[:5]
        cat, reg = key.split("/")[0], region or key.split("/")[1]
        lab = {"region": reg, "hiragana": hira, "plate_kind": kind[cat], "legible": "yes"}
        c = list(cls) + ["<blank>"] * (3 - len(cls))
        for i in range(3):
            lab["class_num_%02d" % (i + 1)] = c[i]
        for i in range(4):
            lab["plate_num_%02d" % (i + 1)] = "<blank>" if num[i] == "." else num[i]
        try:
            idx = [tok[h].index(lab[h]) for h in order]
        except ValueError:
            bad += 1
            continue
        out.append(key + " " + " ".join(str(v) for v in idx))
    with open(os.path.join(crops, "labels.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("%s: %d crops -> labels.txt (%d dropped: token not in spec)" % (crops, len(out), bad))


if __name__ == "__main__":
    main()
