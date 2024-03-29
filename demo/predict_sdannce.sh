#!/bin/bash -e

cd ./2021_07_05_M4_M7
echo "Predicting SDANNCE..."

dannce predict sdannce \
	../../configs/sdannce_rat_config.yaml \
	--dannce-predict-model ../weights/SDANNCE_gcn_bsl_FM_ep100.pth \
	--dannce-predict-dir ./SDANNCE/predict02 \
	--com-file ./COM/predict01/com3d.mat \
	--max-num-samples 500 \
	--batch-size 1

cd ..
echo "DONE!"
