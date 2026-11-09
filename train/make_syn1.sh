#!/bin/sh
# plate_ocr_syn1 を作り直す手順（2026-11-10 に実際に使ったコマンドそのまま）。
# 公開データ（alpr_jp）＋ jp_lpr_image_cpp の合成だけ。会社の実写は使わない。
#
# 置き場所の前提（兄弟ディレクトリ）:
#   ../jp_lpr_image_cpp   合成の生成器（./build.sh で jp_lpr_image をビルド済み）
#   ../yolo_lpr_cpp       学習器（tools/train_ocr.py）と models/plate_ocr_scratch.onnx、spec/labels.txt
#   ../alpr_jp            公開の実写（train/neg/*.jpg も使う）
# 学習は GPU 1 枚で 1 本 5000 step、T4 でおよそ 1 時間。
#
#   sh train/make_syn1.sh gen     # 1. 合成を作る（Windows の書体を地名に使うので Windows で）
#   sh train/make_syn1.sh train   # 2. 種違いで 3 本学習（GPU のある所で。syn/ に 1. の結果を置く）
#   sh train/make_syn1.sh soup    # 3. 3 本の重みを平均 → runs/plate_ocr_syn1.onnx
#   sh train/make_syn1.sh eval runs/plate_ocr_syn1.onnx   # 4. 3 つの評価
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
AZ=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$AZ/.." && pwd)
JP=$ROOT/jp_lpr_image_cpp
YL=$ROOT/yolo_lpr_cpp
ALPR=${ALPR:-$ROOT/alpr_jp}
SYN=${SYN:-$ROOT/syn}
RUNS=${RUNS:-$ROOT/runs}
SPEC=$YL/spec/labels.txt

case "$1" in
gen)
  cd "$JP"
  # 書体の分け方: Kiwi 丸・さわらびゴシックは評価専用（学習に入れない）
  rm -rf out/fonts_train out/fonts_test && mkdir -p out/fonts_train out/fonts_test
  for f in fonts/*.tt*; do
    case $(basename "$f") in KiwiMaru*|Sawarabi*) cp "$f" out/fonts_test/ ;; *) cp "$f" out/fonts_train/ ;; esac
  done
  G="--crops --no-scenes --size 800x600 --width 0.08,0.45 --max-plates 6 --crop-jpg 92 --threads 4"
  # syn_a: 傾いたままの切り抜き
  ./jp_lpr_image --out out/syn_a --count 1900 --seed 1001 --fonts out/fonts_train $G
  # syn_r1: 4 隅から正面向きに切り出した版（解析器の入力と同じ見た目。これが一番効いた）
  ./jp_lpr_image --out out/syn_r1 --count 6000 --seed 4004 --fonts out/fonts_train --crops --no-scenes --rectify \
    --size 1280x960 --width 0.04,0.2 --max-plates 6 --crop-min 56 --crop-jpg 92 --threads 4
  # syn_test: 評価用（未見の書体だけ。Windows の書体も使わない）
  ./jp_lpr_image --out out/syn_test --count 480 --seed 9009 --fonts out/fonts_test --sysfonts none $G
  for s in syn_a syn_r1 syn_test; do python -I "$HERE/jp_crops_to_synth.py" out/$s/crops --spec "$SPEC"; done
  mkdir -p "$SYN"
  for s in syn_a syn_r1 syn_test; do rm -rf "$SYN/$s"; cp -r out/$s/crops "$SYN/$s"; done
  echo "合成: $SYN/{syn_a,syn_r1,syn_test}"
  ;;
train)
  mkdir -p "$RUNS"
  COMMON="--yolo-lpr $YL --init $YL/models/plate_ocr_scratch.onnx --alpr $ALPR
    --neg $HERE/sample_not_plates.tsv --neg-random 3000 --neg-random-glob $ALPR/train/neg/*.jpg --neg-p 0.1
    --commercial-p 0.3 --invert-p 0.1 --batch 64 --lr 3e-4 --warmup 200 --train-bn --backbone-lr-mult 1.0
    --workers 3 --device cuda --steps 5000 --eval-every 500 --eval-limit 0
    --synth $SYN/syn_a --synth $SYN/syn_r1 --synth-region --real-weight 0.4"
  # F: 既定の種（1234）、I: 実写で地名を教える割合 0.2、J: 種 7。GPU が 2 枚なら F と J を並べて回せる
  python "$HERE/train_ocr_real.py" $COMMON --export "$RUNS/F.onnx" --export-last "$RUNS/F_last.onnx"
  python "$HERE/train_ocr_real.py" $COMMON --real-region-p 0.2 --export "$RUNS/I.onnx" --export-last "$RUNS/I_last.onnx"
  python "$HERE/train_ocr_real.py" $COMMON --seed 7 --export "$RUNS/J.onnx" --export-last "$RUNS/J_last.onnx"
  ;;
soup)
  # 最終 step の重み（*_last）ではなく、hold-out で一番良かった版（--export）を平均した
  python "$HERE/soup.py" "$RUNS/plate_ocr_syn1.onnx" "$RUNS/F.onnx" "$RUNS/I.onnx" "$RUNS/J.onnx"
  ;;
eval)
  M=${2:?model.onnx}
  python "$HERE/eval_synth.py" "$M" "$SYN/syn_test" --yolo-lpr "$YL"   # 未見の書体の合成
  python "$HERE/eval_frames.py" "$HERE/video_frames" "$M"              # 動画の 81 コマ
  echo "実写 hold-out: python $HERE/train_ocr_real.py --yolo-lpr $YL --alpr $ALPR --eval-onnx $M"
  echo "動画 8 台: ffmpeg -i IMG_7121.mp4 -f rawvideo -pix_fmt rgb24 - | ./lpr_cli --ocr $M --min-legible 0.5 > r.json && python train/score_video.py r.json"
  ;;
*)
  sed -n 2,14p "$0"; exit 1 ;;
esac
