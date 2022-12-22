#!/bin/bash -e

cd ./2021_07_05_M4_M7
echo "Predicting 3D COM..."

com-predict \
	../../configs/com_mouse_config.yaml \
	--com-predict-weights ../2021_07_06_M3_M6/COM/train02/checkpoint-epoch20.pth \
	--com-predict-dir ./COM/predict02 \
	--max-num-samples 10 \
	--batch-size 1

cd ..
echo "DONE!"
