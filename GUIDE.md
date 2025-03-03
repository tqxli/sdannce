# General Guidance for Custom Experiments

The full pipeline from social video collection to the acquisition and analysis of 3D kinematics encompasses several steps, not all of which are directly supported by s-DANNCE:

1. [Multi-camera video acquisition setup (Campy)](https://github.com/ksseverson57/campy): enable real-time video acquisition and compression for multiple camera streams.

2. [Multi-camera calibration](https://github.com/spoonsso/dannce/tree/master/calibration): determine the camera intrinsics and extrinsics. 

3. [Harvast 3D labels using Label3D GUI](https://github.com/diegoaldarondo/Label3D): if perform fine-tuning and/or training s-DANNCE from scratch on your own data.
   - [Label 3D centers of mass (COM)](#label-3d-coms): if the animals' IDs can be clearly resolved.
   - [Label 3D keypoint poses](#label-3d-keypoint-poses): visualize the Label3D annotations directly within the [GUI](https://github.com/diegoaldarondo/Label3D/blob/master/example.m) or using the [notebook](notebooks/1.visualize_mv_dataset_annotations.ipynb).

4. [Train & predict COM positions](#train--predict-com-positions): if the animals' IDs can be clearly resolved.
5. [Data organization](notebooks/1.visualize_mv_dataset_annotations.ipynb): the multi-camera videos, annotation, io & configuration files should be organized as the demo data.
6. [Finetune or train s-DANNCE from scratch](#fine-tuning-or-training-s-dannce-from-scratch)
   
7. [Inference with trained s-DANNCE models](#inference-using-trained-models)

## Harvest 3D labels
### Label 3D COMs
DANNCE/s-DANNCE performs 3D pose estimation in a **top-down** manner, i.e. it first localizes the animal's approximate position in space and creates a 3D cube enclosing the animal to resolve the 3D posture. As a consequence, the animals' 3D centroid positions must be tracked for all frames, as needed for both training and inference. In Label3D, setting the skeleton profile as `skeletons/com.mat` allows labeling of the animals' centers of mass (COMs). 

For lone animals and social animals with coloring, s-DANNCE supports the training of simple 2D networks for tracking the animals' COM positions using commands `dannce train/predict com`. Please refer to [next section](#train--predict-com-positions) for details. 

:warning: For social animals without markups that prevents ID switches, users must consult more advanced ID tracking methods for resolving the animals' COMs.

### Label 3D keypoint poses
When labeling in Label3D GUI, select the desired skeleton profile under `./skeletons/*.mat` (s-DANNCE uses `rat23` by default) or create a custom profile if use a different landmark/keypoint configuration. Also add this skeleton file in `dannce/engine/skeletons` and keep consistent with the existing ones. Specifically, such a MATLAB file should contain at least
   - `joint_names`: names of N body landmarks.
   - `joints_idx`: [M, 2] indices indicating M connected landmark pairs (e.g. ElbowL - HandL). 1-indexed.

For achieving robust tracking performance in a novel arena setting, the number of complete 3D pose annotations in the training set should at least be about 250-400. Users should avoid annotating temporally contiguous frames to promote diversity in the training samples. This estimated number is respect to individual poses/animals - it is not necessary to label all animals present in each frame. 

## Train & predict COM positions
Users should refer to [`demo/train_com.sh`](demo/train_com.sh) to train a ID-specific COM network and [`demo/predict_com.sh`](demo/predict_com.sh) for performing inference using a pretrained COM model.

## Fine-tuning or training s-DANNCE from scratch

With all preparation ready, sers should refer to the [Training-Notebook](notebooks/3.finetune_on_new_dataset.ipynb) on how to launch a training job. 

To maximize utilization of the manual annotations, we recommend the following steps:

1. Warm up by training the **DANNCE** backbone.
    - Refer to [`demo/train_dannce.sh`](demo/train_dannce.sh) for details.
    - Initialize with RAT7M pretrained weights (`demo/weights/DANNCE_comp_pretrained_r7m.pth`) for better and faster convergence. 
    - Turn on `COM_augmentation: True` and `COM_aug_iters: 2` which augments training samples by 2 times from perturbating COM positions in 3D. 
    - number of epochs: >= 100 or until performance plateau.
2. Start SDANNCE training by finetuning from networks trained in Step 1. 
    - Refer to [`demo/train_sdannce.sh`](demo/train_sdannce.sh) for details.
    - In addition to the supervised L1 loss comparing GT and predicted 3D landmark coordinates, the default training setting applies the bone scale loss (BSL) and the batch consistency loss (BCL), which are both unsupervised. 
        - Turn on `compute_priors_from_data: True` to automatically compute body priors from the trianing labels.
        - Or manually compute the mean and standard deviation of body parts following `joints_idx`, into a [M, 2] numpy array and save as `.npy`. 
    - Turn on `unlabeled_sampling: equal` for training with the aforementioned unsupervised losses. This option samples equal amount of unlabeled samples from the recordings and adds them to the training set. You may also specify `unlabeled_sampling: n`, where n is an integer or float number between [0, 1] indicating the exact number or fraction of unlabeled frames sampled from each experiment.
    - COM augmentation is still highly recommended for achieving the best performance.
    - number of epochs: < 70

## Inference with trained s-DANNCE models
To make 3D pose predictions after a new s-DANNCE model is trained, users should follow the [Inference-Notebook](notebooks/2.inference_using_pretrained_model.ipynb) and adapt correspondingly to the custom dataset and models.

