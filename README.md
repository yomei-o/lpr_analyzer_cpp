# lpr_analyzer_cpp — 動画から通過した車の車番と時刻を一覧にする（ブラウザ / C++）

MP4 を渡して「解析スタート」を押すと、動画を解析して **どの時刻にどのナンバーの車が通過したか** を
一覧にします。MP4 の読み込み、H.264 のデコード、ナンバーの検出・認識・追跡、枠と数字の描画まで
すべて C++ で、ブラウザでは WebAssembly で動きます（動画はどこにも送信されません）。

## ▶ デモ

**https://yomei-o.github.io/lpr_analyzer_cpp/**

サンプル動画 `IMG_7121.mp4`（交差点、19 秒、1080p）が最初から選ばれているので、動画を持っていなくても
そのまま「解析スタート」で試せます。自分の MP4 はドロップかクリックで一覧に追加して選びます。

サンプルの結果（Edge でページを動かしたもの。`node tests/test_node.js` も同じ 8 台）:

| 時刻 | ナンバー | 確度（4 桁） | 読んだ回数 |
|---|---|---|---|
| 00:00.0 | ・・・1 | 94% | 8 |
| 00:02.0 | 12-09 | 100% | 21 |
| 00:06.0 | ・・・9 | 92% | 17 |
| 00:09.4 | 33-00 | 60% | 5 |
| 00:10.0 | ・・・6 | 83% | 10 |
| 00:14.0 | 96-99 | 60% | 2 |
| 00:14.0 | 22-56 | 95% | 16 |
| 00:15.6 | 13-08 | 93% | 8 |

4 桁の数字は高い確度で読めます。地名・分類番号・ひらがなは表示していますが、遠いナンバーではまだ誤りが多めです
（例: 江戸川 → 北九州）。

## 解析のしかた

```
MP4 ─ mediakit Mp4Reader（FFmpeg の demux）─ VideoDecoder（ブラウザの WebCodecs / 無ければ FFmpeg）─ RGB
   └→ lpr_analyzer（src/lpr_analyzer.hpp）
        全体スキャン   2 秒ごと（0.5 fps）: フレーム全体を yolov8n 640 で検出 → 新しいナンバーを見つける
        追跡中の再解析 0.2 秒ごと: 追跡中のナンバーの予測位置のまわりの正方形の窓だけを yolov8n 320 で検出
        読み取り      4 隅補正 + 位置別の分類器（yolo_lpr_cpp）
        トラッカー    予測位置との距離で対応付け、フレームごとの各桁の確率を足し込んで多数決
        確定          1 秒見えなくなったら「通過した」として一覧に入れる（時刻 = 最初に見えた時刻）
        描画          枠・4 桁の数字・探索窓をフレームに C++ で描いてページに渡す
```

- 1 フレームの読み取りに頼らず、追跡中に何度も読んだ確率を足し込むので、1 回だけの読み違いに引きずられません。
- 窓だけを軽い検出器で見るので、追跡中の再解析は全体スキャンの 1/4 程度の時間で済みます。
- 一覧に入るのは、4 桁の平均確度が 0.55 以上、かつ 2 回以上読めたもの（1 回だけなら 0.85 以上）。
  看板などの誤検出は追跡を始めないので一覧に出ません。
- 見失って同じナンバーを数秒以内に見つけ直したときは、1 台にまとめます。

ページの設定で、全体スキャンの間隔（1 / 2 / 4 秒）、再解析の間隔（0.1 / 0.2 / 0.5 秒）、全体スキャンの
検出器（640 / 320）を変えられます。結果は CSV で保存できます。

## ファイル

```
src/lpr_analyzer.hpp      解析の本体（検出・読み取り・追跡・確定・描画）。OS に依存しない
src/analyzer_wasm.cpp     ブラウザ版: MP4 → デコード → 解析を az_step() で 1 歩ずつ進める
src/analyzer_cli.cpp      ネイティブ版のテスト用ドライバ（ffmpeg が出す生 RGB を標準入力で受ける）
docs/                     GitHub Pages（index.html, worker.js, analyzer.js/.wasm, models/, samples/）
tests/test_node.js        node で wasm をそのまま動かす回帰テスト
cmake/ffmpeg_min/         FFmpeg のコーデック一覧を MP4 の読み込みに要るものだけに絞ったもの
third_party/mediakit/     mediakit_cpp の codec 層（FFmpeg 7.1.1 を含む）と common
third_party/yolo_lpr/     yolo_lpr_cpp の推論エンジン（pure/）とラベル定義（spec/labels.txt）
```

ページはメインスレッドを止めないよう、wasm を Web Worker で動かしています。WebCodecs は非同期にフレームを
返すので、Worker は `az_step()` を呼んでは一瞬イベントループに戻る、を繰り返します。

## ビルド

ブラウザ版（emsdk は `C:\prog\emsdk\emsdk` にある前提。`docs/analyzer.js` と `docs/analyzer.wasm` ができる）:

```sh
cmake -S . -B build-wasm -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=C:/prog/emsdk/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
cmake --build build-wasm
node tests/test_node.js                      # サンプル動画で回帰テスト（期待するナンバーが全部出るか）
cd docs && python -m http.server 8000        # http://localhost:8000/
```

ネイティブ版（解析部分だけ。デコードは ffmpeg に任せる）:

```sh
g++ -std=c++20 -O2 -Isrc -Ithird_party/yolo_lpr/pure -Ithird_party/yolo_lpr/pure/third_party \
    src/analyzer_cli.cpp -o lpr_cli.exe
ffmpeg -i docs/samples/IMG_7121.mp4 -f rawvideo -pix_fmt rgb24 - | ./lpr_cli.exe --size 1920x1080 --fps 30 --out out
```

`--out` を付けると、解析したフレームを描画したもの（`out/fNNNNN.png`）と、各結果のプレートと場面の画像を書き出します。

## 速度

サンプル（19 秒）で、全体スキャン 10 回 + 追跡中の再解析 81 回 = 91 フレームを解析します（スレッド無し、CPU 1 コア）。

| | 全体スキャン 1 回 | 再解析（窓 1 つ） | 全体 |
|---|---|---|---|
| Edge（wasm + WebCodecs） | 約 2.5〜4 秒 | 約 0.5〜0.7 秒 | 計算 110 秒、実時間 約 130 秒 |
| node（wasm + FFmpeg ソフトデコード） | 約 5 秒 | 約 0.7〜1 秒 | 計算 173 秒 |
| ネイティブ（g++ -O2、`lpr_cli`） | 約 3 秒 | 約 0.6〜0.8 秒 | 計算 117 秒 |

急ぐときは「ナンバーを見つけた後の再解析」を 0.5 秒にするか、全体スキャンの検出器を 320 にしてください。

解析の中身は OS に依存しないので、同じ `lpr_analyzer.hpp` を GPU のある PC で動かせば、そのまま高速なサーバ版になります。
その場合、推論は yolo_lpr_cpp の GPU ビルド、デコードは mediakit のネイティブ版（D3D11VA / VideoToolbox などのハードウェアデコード）に任せます。

## ライセンス

- このリポジトリのコード（`src/`, `docs/`, `tests/`）: 未定（LICENSE ファイルはまだ置いていない）
- `third_party/yolo_lpr/`: yolo_lpr_cpp（BSD 3-Clause、`third_party/yolo_lpr/LICENSE` と `THIRD_PARTY_NOTICES.md`）。モデルも同リポジトリのもの
- `third_party/mediakit/codec/ffmpeg/`: FFmpeg 7.1.1（LGPL 2.1 以降）。mediakit_cpp が平坦化したもの
- `third_party/mediakit/`: mediakit_cpp。`VideoDecoder::poll()` を 1 つ足してある（WebCodecs の非同期デコードを待つため）
- `docs/samples/IMG_7121.mp4`: テスト用のサンプル動画
