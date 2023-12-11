# Running Demo
## Download Data
1. Follow [this Box link](https://duke.box.com/s/2aw5r4hb3u57p1abt99n15f6hkl36x5k) and download the demo data (~5 GB) to the directory `demo` as `demo_data.zip`.

2. In the command line, change the current directory to `demo` by `cd demo` and run the script `sh prepare_demo.sh` to unzip and prepare the demo data. The final directory should appear as:
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

## Quickest demo of s-DANNCE
### Prediction with s-DANNCE
Here, using a pretrained s-DANNCE model (`demo/weights/SDANNCE_gcn_bsl_FM_ep100.pth`), we predict social animal 3D poses for the first 10 frames of an experiment `2021_07_05_M4_M7` by running the script
```
sh predict_sdannce.sh
```

The prediction results should be saved into `demo/2021_07_05_M4_M7/SDANNCE/predict02`. 

### Visualize predictions
We then visualize the predictions just obtained from last step, from running
```
sh vis_sdannce.sh
```
You should find the resulted video overlay with keypoint projections in `demo/2021_07_05_M4_M7/SDANNCE/predict02/vis/*.mp4`.


<details>
<summary>MovieWriter stderr during visualization</summary>
- "MovieWriter stderr:
[libopenh264 @ 0x55b92cb33580] Incorrect library version loaded
Error initializing output stream 0:0 -- Error while opening encoder for output stream #0:0 - maybe incorrect parameters such as bit_rate, rate, width or height ..."

Try update the ffmpeg version by `conda update ffmpeg`.
</details>


## General use of s-DANNCE
The users are highly encouraged to refer to the provided `demo/*.sh` scripts and explore the functionality.

### Train & Predict Animal Centroids
DANNCE/s-DANNCE performs 3D pose estimation in a **top-down** manner, i.e.
- first localizes the animal's approximate position in space
- creates a 3D cube enclosing the animal and resolves the associated 3D posture within.

For the first step, we start by training a center-of-mass (COM) prediction network that simultaneously detects both animal. In the provided videos, the paired animals were painted respectively with blue and red colors to avoid ID switches.

To train such a model, one can modify and run
```
sh train_com.sh
```
For labeling COMs on your custom data, please refer to [Label3D](https://github.com/diegoaldarondo/Label3D). 

We can predict COMs using a trained COM model, by running `demo/predict_com.sh`. In the previous demo, we directly used a set of presaved COM predictions (`demo/2021_07_05_M4_M7/COM/predict01/com3d.mat`).

### Train Pose Estimation Models
With the 3D COMs predicted from the last step, we can start training a DANNCE/s-DANNCE model for extracting animal 3D poses. 

As a quick demonstration, we can directly run `demo/train_sdannce.sh` to start training a s-DANNCE pose estimation model in `demo/2021_07_06_M3_M6/SDANNCE/train02`.

To avoid confusion, the ground truth labels used for training are stored in `*Label3D*.mat` files under each directory, e.g., `demo/2021_07_06_M3_M6/ANNOT_Label3D_B_dannce.mat`, obtained from Label3D. The set of experiments used for training is specified like in `demo/2021_07_06_M3_M6/io.yaml`, as
```
exp:
    - label3d_file: '../2021_07_06_M3_M6/ANNOT_Label3D_B_dannce.mat'
      com_file: '../2021_07_06_M3_M6/COM/predict01/instance0com3d.mat'
    - label3d_file: '../2021_07_06_M3_M6/ANNOT_Label3D_R_dannce.mat'
      com_file: '../2021_07_06_M3_M6/COM/predict01/instance1com3d.mat'
```
which are then retrieved along with the video frames and combined for training.


