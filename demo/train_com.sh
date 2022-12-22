#!/bin/bash -e

cd ./2021_07_06_M3_M6
echo "Training COM..."

com-train ../../configs/com_mouse_config.yaml

cd ..
echo "DONE!"
