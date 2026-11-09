"""同じ出発点から学習した認識器（ONNX）の重みを平均する（model soup）。

  python train/soup.py out.onnx a.onnx b.onnx [c.onnx ...]

グラフは最初のファイルのもの。浮動小数の initializer（畳み込み・BN の統計を含む）を要素ごとに平均する。
同じ出発点（plate_ocr_scratch）から同じレシピで学習したものどうしでないと意味がない。
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    models = [onnx.load(p) for p in ins]
    base = models[0]
    others = [{t.name: numpy_helper.to_array(t) for t in m.graph.initializer} for m in models[1:]]
    n = 0
    for i, t in enumerate(base.graph.initializer):
        a = numpy_helper.to_array(t)
        if a.dtype.kind != "f":
            continue
        acc = a.astype(np.float64)
        for o in others:
            acc = acc + o[t.name].astype(np.float64)
        base.graph.initializer[i].CopyFrom(numpy_helper.from_array((acc / len(models)).astype(a.dtype), t.name))
        n += 1
    onnx.save(base, out)
    print("%s: averaged %d tensors over %d models" % (out, n, len(models)))


if __name__ == "__main__":
    main()
