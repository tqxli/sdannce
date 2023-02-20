#!/bin/bash -e

echo "Preparing test data..."

# unzip demo_data.zip
# mv demo_data/* .
# rm demo_data.zip
# rm demo_data

cp test_files/com_io.yaml 2021_07_06_M3_M6/io.yaml
cp test_files/com_io.yaml 2021_07_05_M4_M7/io.yaml
cp test_files/test_social_com_dannce.mat 2021_07_06_M3_M6/test_social_com_dannce.mat
cp test_files/test_social_dannce_dannce.mat 2021_07_06_M3_M6/test_social_dannce_dannce.mat
echo "DONE!"