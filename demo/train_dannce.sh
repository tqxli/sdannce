#!/bin/bash -e

cd ./2021_07_06_M3_M6
echo "Training DANNCE..."

dannce train dannce \
	../../configs/dannce_rat_config.yaml \
	--train-mode finetune \
	--dannce-finetune-weights ../weights/DANNCE_comp_pretrained_single+r7m.pth \
	--net-type compressed_dannce \
	--use-npy True

cd ..
echo "DONE!"
