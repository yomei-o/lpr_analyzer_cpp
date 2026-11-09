"""デモ動画 IMG_7121 の解析結果（analyzer_cli の標準出力の JSON）を、目視で作った正解と照らし合わせる。

  python train/score_video.py result.json [more.json ...]

正解は元の 1920x1080 のコマを拡大して読んだもの。33-00（軽の事業用、黒地）は地名・分類番号・ひらがなが
元の画質でも読めないので、4 桁だけを採点する。
"""
import json
import sys

TRUTH = {                     # number: (region, class, hiragana)
    "・・・1": ("松戸", "309", "ま"),
    "12-09": ("江戸川", "310", "そ"),
    "・・・9": ("葛飾", "300", "す"),
    "33-00": (None, None, None),
    "・・・6": ("葛飾", "200", "さ"),
    "96-99": ("足立", "500", "え"),
    "22-56": ("足立", "580", "ふ"),
    "13-08": ("足立", "331", "も"),
}


def score(path):
    rs = json.load(open(path, encoding="utf-8"))
    got = {r["number"]: r for r in rs}
    found = sum(1 for k in TRUTH if k in got)
    extra = [r["text"] for r in rs if r["number"] not in TRUTH]
    ok = {"region": 0, "cls": 0, "hira": 0}
    n = 0
    rows = []
    for k, (reg, cls, hira) in TRUTH.items():
        r = got.get(k)
        if reg is None:
            rows.append("  %-6s %-24s (4 桁だけ)" % (k, r["text"] if r else "---"))
            continue
        n += 1
        if not r:
            rows.append("  %-6s %-24s 見逃し" % (k, "---"))
            continue
        marks = []
        for key, want in (("region", reg), ("cls", cls), ("hira", hira)):
            good = r.get(key) == want
            ok[key] += good
            marks.append("o" if good else "x")
        rows.append("  %-6s %-24s 地名%s 分類%s かな%s  (正解 %s %s %s)" % (k, r["text"], *marks, reg, cls, hira))
    print("%s: %d/%d 台, 誤り %d 件 | 地名 %d/%d  分類番号 %d/%d  ひらがな %d/%d"
          % (path, found, len(TRUTH), len(extra), ok["region"], n, ok["cls"], n, ok["hira"], n))
    for row in rows:
        print(row)
    for e in extra:
        print("  余分: %s" % e)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        score(p)
