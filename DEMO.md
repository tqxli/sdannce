# Running Demo
## :arrow_down: Download Data
1. Follow [this Box link](https://duke.box.com/s/2aw5r4hb3u57p1abt99n15f6hkl36x5k) and download the demo data (~5 GB) to the directory `demo` as `demo_data.zip`.

2. From the command line, change the current directory to `demo` by `cd demo` and run the script `sh prepare_demo.sh` to unzip and prepare the demo data. The final directory should appear as:
    ```
    demo
        /2021_07_05_M4_M7
        /2021_07_06_M3_M6
        /2021_07_07_M5_M1
        /SCN2A_WK1_2022_09_16_M1
        /SCN2A_SOC1_2022_09_23_M1_M6
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

## :zap: Quickest Demo of s-DANNCE Functionality

Here, we demonstrate quick inference using a pretrained s-DANNCE model and visualization of the tracking results. Please checkout the notebook [`./notebooks/2.inference_using_pretrained_model.ipynb`](notebooks/2.inference_using_pretrained_model.ipynb) or run the following bash scripts under `demo/*.sh`:

### Prediction with s-DANNCE
Using a pretrained s-DANNCE model (`demo/weights/SDANNCE_gcn_bsl_FM_ep100.pth`) to predict the social animal 3D poses for the first 500 frames of an experiment `2021_07_05_M4_M7` by running the script
```
sh predict_sdannce.sh
```
The prediction results should be saved into `demo/2021_07_05_M4_M7/SDANNCE/predict02`. 

### Prediction visualization
We then visualize the predictions just obtained from last step, from running
```
sh vis_sdannce.sh
```
You should find a 10-second video overlay with keypoint projections in `demo/2021_07_05_M4_M7/SDANNCE/predict02/vis/*.mp4`.


<details>
<summary> In case that you encountered a MovieWriter stderr during visualization</summary>
- Error: "MovieWriter stderr:
[libopenh264 @ 0x55b92cb33580] Incorrect library version loaded
Error initializing output stream 0:0 -- Error while opening encoder for output stream #0:0 - maybe incorrect parameters such as bit_rate, rate, width or height ..."

Try update the ffmpeg version by `conda update ffmpeg`.
</details>


## :bulb: Unpack s-DANNCE Command Line Usage
The basic usage of the s-DANNCE CLI is as follows:

```
dannce <command> [<mode>]
```

where the available commands are:

- `train`: Train the network model.
- `predict`: Predict using the network model.
- `predict-multi-gpu`: Predict using the network model on multiple GPUs.
- `merge`: Merge network prediction results.

Run `dannce <command> --help` to see the available options for each command.

For launch training & prediction jobs, respectively run
```
dannce train <mode> [<args>]
dannce predict <mode> [<args>]
```

The available modes are:

- `com`: Train a center-of-mass (COM) network.
- `dannce`: Train a DANNCE network.
- `sdannce`: Train the social DANNCE network.

For more details of the CLI setups, check out [cli.md](dannce/cli.md).

For specific examples on using these CLI commands, check out different bash scripts `demo/train*.sh` and `demo/predict*.sh`. 


### :question: Why COM
DANNCE/s-DANNCE performs 3D pose estimation in a **top-down** manner, i.e. it first localizes the animal's approximate position in space and creates a 3D cube enclosing the animal to resolve the 3D posture. Thus, the estimation of animals' centroids, or centers of mass (COMs) must be done prior to any DANNCE/s-DANNCE training.

## :unlock: What's Next
Check out [`GUIDE.md`](GUIDE.md) for how to set up for custom data collection and experiments. 

