#!/bin/sh
set -e
cd "$(dirname "$0")/.."
out=models/hand_landmarker.task
if [ -f "$out" ]; then
  echo "already have $out"
  exit 0
fi
mkdir -p models
curl -fsSL -o "$out.part" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
mv "$out.part" "$out"
echo "got $out"
