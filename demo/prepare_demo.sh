#!/bin/bash -e

echo "Preparing demo data..."

unzip demo_data.zip
mv demo_data/* .
rm demo_data.zip
rm demo_data

echo "DONE!"