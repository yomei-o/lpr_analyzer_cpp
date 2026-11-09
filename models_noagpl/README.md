# AGPL と以前のモデルの重みを使わないモデル

ライセンス: **BSD 3-Clause**（リポジトリの `LICENSE`）。

デモの「自前学習」で使っているモデル（`docs/models_bsd/` と同じファイル）。yolo_lpr_cpp の元モデルは、検出器が
Ultralytics の `yolov8n.pt` から転移学習したもの（AGPL-3.0）、認識器が lpr_cpp の以前のモデルの重みから
学習したもの。ここにあるのは、どちらも使わずに**ランダム初期化から**学習した。学習は 2026-11-08〜09、Kaggle の T4。

| ファイル | 何か |
|---|---|
| `../docs/models_bsd/plate_det_scratch_640.onnx` / `_320.onnx` | 検出器（yolov8n 構造、nc=1）。同じ重みを 640 と 320 の入力で |
| `../docs/models_bsd/plate_ocr_syn1.onnx` | 認識器（11 head。最後の legible head が「ナンバーか」）。下の「認識器 plate_ocr_syn1」 |
| `../docs/models_bsd/plate_ocr_scratch.onnx` | 前の認識器（syn1 の出発点） |
| `plate_det_scratch_640.onnx` / `_320.onnx` / `plate_ocr_scratch.onnx` | 1 回目の版（合成データだけ。記録用） |
| `plate_det_scratch_best.pt` | 1 回目の検出器の PyTorch 重み（微調整の出発点） |
| `plate_det_mix_best.pt` / `det_mix_train_log.csv` | デモの検出器の PyTorch 重み（微調整 40 epoch の best = epoch 33）と学習ログ |
| `det_train_log.csv` | 1 回目の検出器の学習ログ |

## どう作ったか

**検出器**（`train/plate_det_torch.py`。YOLOv8n を PyTorch で手書き、Ultralytics のコードは使っていない）

1. 合成 2 万フレーム（`jlpr gen-det`：alpr_jp の実ナンバー切り抜きを alpr_jp の背景に貼ったもの）で 100 epoch
2. 合成 9000 ＋ **Open Images V7 の Vehicle registration plate 5362 枚**（CC BY、`train/fetch_openimages_plates.py`）
   ＋ **alpr_jp の車の写真 224 枚**（このモデルで候補を出し、目視で確かめた箱 `train/sample_plates.tsv`）で微調整
   - 実写の写真だけで微調整すると、その 1 台の固定カメラの場面に寄ってしまい、サンプル動画では悪化した（2〜3 台）。
     世界中の実写場面（Open Images）を混ぜて解決した
3. グラフは `jlpr init-det` が書き、重みを `train/export_onnx.py` で名前どおりに入れる。PyTorch と ONNX の出力一致を確認済み

**認識器**（`train/train_ocr_real.py`。yolo_lpr_cpp の `tools/train_ocr.py` を、ランダム初期化 ＋ 下の 3 点で使う）

1. **実写の全 head ラベル**: alpr_jp の 1273 枚（地名フォルダの 720 枚 ＋ 地名フォルダの無い 軽/事業用 553 枚）の
   分類番号・ひらがな・4 桁・地名を目視で書き起こした（`train/alpr_labels.tsv`, `alpr_labels_flat.tsv`）。
   元の学習は実写に地名のラベルしか無く、数字は合成の書体からしか学べないので、ゼロからだと実写の 4 桁がほぼ読めなかった（3.5%）
2. **負例**: 検出器の候補のうち目視で却下した 960 個（`train/sample_not_plates.tsv`：反射板・コーン・看板・ステッカーなど）と
   背景のランダムな切り抜き 3000 個を legible = no として学習。解析器は legible < 0.5 の検出を捨てる
3. **濃色地のナンバー**: 事業用（緑地白字）・軽事業用（黒地黄字）を多めに引き、色反転の拡張も入れる

合成データは GenSenRounded だけで作った（下の「見つけた不具合」）。PlateYOLO-JP（AGPL）が付けた疑似ラベルは使っていない。

## 測った結果

認識器（alpr_jp の hold-out、正解は手で書き起こしたラベル）:

| | 地名 | 分類番号 | ひらがな | 4 桁すべて正解 |
|---|---|---|---|---|
| 元モデル `plate_ocr_v7_bal`（削除済み、自家用 144 枚） | **97.9%** | 98-99% | 79.7% | 97.9% |
| **自前（自家用 144 枚）** | 88.2% | 94-99% | **95.5%** | **99.3%** |
| 元モデル（軽・事業用 111 枚） | **97.2%** | 93-98% | 87.5% | 93.7% |
| **自前（軽・事業用 111 枚）** | 92.5% | **98-100%** | **92.3%** | **98.2%** |

元モデル（lpr_cpp の重み由来、2026-11-09 に削除）の数字は削除前の記録。地名だけは元モデルに届かなかった（alpr_jp の実写は 63 地名しか無く、残りは合成の書体からしか学べない）。

検出器（`data/det_eval` 合成 150 フレーム）: 元 640 は recall 97.1% / mAP50 0.982、1 回目の自前 640 は 96.0% / 0.984。
実写の場面（Open Images の検証 719 枚 ＋ alpr_jp 31 枚）では微調整後 AP50 0.83。

サンプル動画 IMG_7121（実在の 8 台）: **自前のモデルだけで 8 台すべて・誤り 0 件**（Edge でもネイティブでも）。元モデルはネイティブ 8 台、Edge 7 台。

## 認識器 `plate_ocr_syn1`（2026-11-10〜 既定）

`plate_ocr_scratch` は地名を alpr_jp にある 63 地名でしか覚えておらず、サンプル動画（東京東部の車）の地名を 1 台も読めなかった。
間違えた先はどれも alpr_jp の地名だった（葛飾→長崎、足立→広島）。そこで、全 138 地名を描ける合成のナンバーで学習し直した。

1. 合成: [jp_lpr_image_cpp](https://github.com/yomei-o/jp_lpr_image_cpp) で 2 セット
   - 傾いたままの切り抜き 6292 枚（`--width 0.08,0.45`）
   - **4 隅から正面向きに切り出した版 18819 枚**（`--rectify --width 0.04,0.2 --crop-min 56`）。解析器は 4 隅の補正をしてから読むので、
     その入力と同じ見た目にする。これが一番効いた（傾いた切り抜きだけ、小さく潰れた版を足す、は効かないか逆効果だった）
   - 書体は学習用 18 種（同梱 6 ＋ Windows の 12。Windows の書体は地名だけ）。Kiwi 丸・さわらびゴシックは評価用に学習から外した
   - `train/jp_crops_to_synth.py` で学習器の合成セット形式に変換
2. 学習: `train/train_ocr_real.py`（`plate_ocr_scratch.onnx` から、alpr_jp 実写＋負例＋上の合成、`--synth-region --real-weight 0.4
   --lr 3e-4 --steps 5000`）を種違いで 3 本（うち 1 本は `--real-region-p 0.2`）
3. 3 本の重みを平均（`train/soup.py`）。単体は動画 81 コマの地名が 42〜46%、平均すると 50.6%

実際に使ったコマンド（生成の種・書体の分け方・学習の引数・平均した 3 本）は `train/make_syn1.sh` にそのまま書いてある。

評価は 3 つ。どれも学習に使っていない。

- alpr_jp の hold-out（`--eval-onnx`）
- 見たことの無い書体の合成 1605 枚（`train/eval_synth.py`、Kiwi 丸・さわらびゴシックだけで描いたもの）
- サンプル動画: 解析器が認識器に渡した入力 81 コマ（`train/eval_frames.py train/video_frames`、`lpr_cli --dump` で作成）と、
  動画全体の 8 台（`train/score_video.py`、正解は元の 1920x1080 のコマを目視）

| | 実写 地名（自家用 / 軽・事業用） | 実写 ひらがな | 実写 4 桁 | 合成（未見の書体）地名 / alpr_jp に無い地名 | 動画 81 コマ 地名 | 動画 8 台 |
|---|---|---|---|---|---|---|
| `plate_ocr_scratch` | 88.2% / 92.5% | 95.5% / 92.3% | 99.3% / 98.2% | 33.9% / 23.5% | 4.9% | 地名 0/7・分類 4/7 |
| `plate_ocr_ft1` | 92.4% / 92.5% | 96.2% / 93.3% | 99.3% / 99.1% | 32.7% / 22.4% | 24.7% | 地名 2/7・分類 5/7 |
| **`plate_ocr_syn1`** | 91.7% / **94.3%** | 96.2% / **94.2%** | **100% / 100%** | **86.6% / 84.5%** | **50.6%** | **地名 5/7・分類 7/7・誤り 0** |

残っている誤り: 葛飾 300 す（→倉敷）と、ぼけた 足立 331 も（→熊本 む）。

## 見つけた不具合（yolo_lpr_cpp 側）

どれも yolo_lpr_cpp 本体で直した（2026-11-09、`63135f7`）。Droid の書体には無い字だけ GenSen から借りて描くようにし、
`--alpr-labels spec/alpr_labels.tsv` で下の 553 枚と全 head のラベルが C++ / Python 両方の学習器で使えるようになった。
ついでに、C++ の `read_alpr` の「軽」が簡体字「轻」のバイト列になっていたのも見つけて直した。

- `fonts/DroidSansFallbackFull.ttf` には ASCII の数字が無い。`spec/fonts.txt` に入っているので、`jlpr gen` / `gen-det` の
  合成のおよそ半分で 4 桁と分類番号が豆腐（□）で描かれる
- alpr_jp の `自家用(軽)/` `事業用/` `事業用(軽)/` は地名のフォルダに分かれていない（ファイルが直に置かれている）ので、
  `tools/train_ocr.py` の `AlprSet` はこの 553 枚を読まない。濃色地のナンバー（事業用）は実写で 1 枚も学習されていなかった
- alpr_jp の `自家用/大分/` に長崎ナンバーが 2 枚入っている（`train/alpr_labels.tsv` の region 列で訂正）

## クレジット

- Open Images V7（Google）: アノテーション CC BY 4.0、画像 CC BY 2.0（各画像の作者は Open Images のメタデータにある）
- dyama/alpr_jp: MIT（画像の著作権は撮影者）
