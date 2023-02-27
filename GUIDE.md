# Training Tips
For training new networks FROM SCRATCH on custom data:

## Training Preparation
- Please refer to the original DANNCE instructions for [camera calibration](https://github.com/spoonsso/dannce/tree/master#using-dannce-on-your-data) of the multi-view camera setup and use [Label3D](https://github.com/diegoaldarondo/Label3D) for annotating both COM and animal poses. 
    - Refer to `./demo` for organizing the data and associated datafiles.
- The size of the final training set, i.e., the number of complete 3D pose annotations, should ideally be about 250-400 (or more). 
    - Avoid annotating temporally contiguous frames to promote diversity in the training samples.
    - This number is respect to individual poses/animals - not necessary to label all animals present in each frame. 
- If use a different landmark/keypoint configuration, add corresponding skeleton file in `dannce/engine/skeletons` and keep consistent with the existing ones. Specifically, such a MATLAB file should contain at least
    - `joint_names`: names of N body landmarks.
    - `joints_idx`: [M, 2] indices indicating M connected landmark pairs (e.g. ElbowL - HandL). 1-indexed.

## Training Workflow
To maximize utilization of the manual annotations, we recommend the following steps:

1. Warm up by training the **DANNCE** encoder-decoder first. Refer to `demo/train_dannce.sh`. 
    - Initialize with RAT7M pretrained weights (`demo/weights/DANNCE_comp_pretrained_r7m.pth`) for better and faster convergence. 
    - If time permits, we recommend using `COM_augmentation: True` which augments training samples by perturbating COM positions in 3D. 
    - number of epochs: < 300 or until performance plateau.
2. Start SDANNCE training by finetuning from networks trained in 1. Refer to `demo/train_sdannce.sh` for details.
    - In addition to the supervised L1 loss over GT and predicted 3D landmark coordinates, the default training setting applies the bone scale loss (BSL) and the batch consistency loss (BCL), which are both unsupervised. 
        - To apply the bone length loss, compute the mean and standard deviation of body parts, as specified `joints_idx` above, into a [M, 2] numpy array and save as `.npy`. 
    - We recommend turn on `unlabeled_sampling: equal` for training with the aforementioned unsupervised losses. This option samples equal amount of unlabeled samples from the recordings and adds them to the training set. 
    - COM augmentation is still highly recommended for achieving the best performance.
    - number of epochs: < 50

## Tips
- Turn on `use_npy: True` for training if memory is limited and/or the size of training set is large - this option saves training volumes to disk as `.npy` instead of maintaining them in the memory.