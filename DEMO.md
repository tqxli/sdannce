# Running Demo
## Download Demo Data
1. Follow [this Box link](https://duke.box.com/s/2aw5r4hb3u57p1abt99n15f6hkl36x5k) and download to the directory `demo` as `demo_data.zip`.

2. Run `sh prepare_demo.sh`. The final directory should appear as:
```
demo
    /2021_07_05_M4_M7
    /2021_07_06_M3_M6
    /2021_07_07_M5_M1
    /visualization
    /weights
    predict_com.sh
    predict_sdannce.sh
    prepare_demo.sh
    train_com.sh
    train_dannce.sh
    train_sdannce.sh
    vis_sdannce.sh

```

## Predict Social Animal Centroids
To infer the animals' centroids/COMs in the 3D space, run
```
sh predict_com.sh
```
The COMs of both animals are saved into `2021_07_05_M4_M7/COM/predict02/com3d.mat`and will be used next to anchor volumetric representations.

## Predict Social Animal Poses
To generate the 3D poses, run
```
sh predict_sdannce.sh
```
Both animals' 3D poses are saved into `demo/2021_07_05_M4_M7/SDANNCE/predict02`.

## Visualize predictions
To visualize the predictions we just obtained from last step, run
```
sh vis_sdannce.sh
```
You should find the resulted video in `demo/2021_07_05_M4_M7/SDANNCE/predict02/vis`.
