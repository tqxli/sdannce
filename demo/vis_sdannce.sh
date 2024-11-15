#!/bin/bash -e

echo "Generating SDANNCE prediction video..."

ROOT=2021_07_05_M4_M7
PRED=SDANNCE/predict02

python ../dannce/engine/utils/vis.py --root $ROOT --pred $PRED  --cameras 3 --n_frames 500

echo "Saved to $ROOT/$PRED/vis"