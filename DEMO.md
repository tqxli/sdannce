# Running Demo
## Download Data
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

## Train & Predict Animal Centroids
DANNCE/s-DANNCE performs 3D pose estimation in a top-down manner, i.e. first localizes the animal's approximate position in space and then resolves the associated 3D posture. We start by training a center-of-mass (COM) prediction network that simultaneously detects both animal. In this set of videos, the two animals were painted respectively with blue and red colors to avoid ID switches.

To train such a model, run
```
sh train_com.sh
```
Given the limited amount of COM labels in this small demo, we will use presaved COM predictions (`demo/2021_07_05_M4_M7/COM/predict01/com3d.mat`) to proceed with the next part of this demo. For labeling on the custom data, refer to [Label3D](https://github.com/diegoaldarondo/Label3D). 

To predict COMs using a pretrained model, please refer to `demo/predict_com.sh`.

## Train & Predict Social Animal Poses
To train a SDANNCE pose estimation model, refer to `demo/train_sdannce.sh`.

Here, using a pretrained SDANNCE model (`demo/weights/SDANNCE_gcn_bsl_FM_ep100.pth`), run the following to predict the associated 3D poses:
```
sh predict_sdannce.sh
```
The prediction results are saved into `demo/2021_07_05_M4_M7/SDANNCE/predict02`.

## Visualize predictions
To visualize the predictions just obtained from last step, run
```
sh vis_sdannce.sh
```
You should find the resulted video in `demo/2021_07_05_M4_M7/SDANNCE/predict02/vis/*.mp4`.
