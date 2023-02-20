
# DANNCE CLI

The DANNCE CLI provides a command-line interface for the DANNCE library. It allows users to train and predict with the DANNCE model, as well as merge prediction results.

# Usage

## Basic Usage

The basic usage of the DANNCE CLI is as follows:

```
dannce <command> [<mode>]
```

The available commands are:

- `train`: Train the network model.
- `predict`: Predict using the network model.
- `predict-multi-gpu`: Predict using the network model on multiple GPUs.
- `merge`: Merge network prediction results.

To see the available options for each command, run:

```
dannce <command> --help
```

# Train

Train the DANNCE model.

```
dannce train <mode> [<args>]
```

The available modes are:

- `com`: Train the a center-of-mass network.
- `dannce`: Train a DANNCE network.
- `sdannce`: Train the social DANNCE network.

To see the available options for each mode, run:

```
dannce train <mode> --help
```

# Predict

Predict using the DANNCE model.

```
dannce predict <mode> [<args>]
```


To see the available options, run:

```
dannce predict <mode> --help
```

# Predict-Multi-GPU

Predict across multiple GPUs using the model.

```
dannce predict-multi-gpu <mode> [<args>]
```



To see the available options, run:

```
dannce predict-multi-gpu <mode> --help
```


# Merge

Merge prediction results.

```
dannce merge [<args>]
```


To see the available options, run:

```
dannce merge --help
```


# Examples

## Training in COM Mode

To train a center-of-mass network, run:

```dannce train com com_config.yaml```

You can also set hyperparameters from the command-line:

```dannce train com com_config.yaml --epochs=2```


## Predicting

To predict using a DANNCE model, run:

```dannce predict dannce dannce_config.yaml```

## Multi-gpu prediction

To predict across multiple gpus, you'll need access to a slurm cluster. Additionally you'll need to specify the slurm submission parameters for your institution using a `.yaml` file. The path to this file should be included in your configuration file under the key `slurm_config`.

```dannce predict-multi-gpu dannce dannce_config.yaml```


### Merging Prediction Results

To merge DANNCE prediction results, run:

```dannce merge dannce_config.yaml```

# Slurm

If you are using a slurm cluster, you can submit training and prediction jobs to the cluster using the `--sbatch` command line argument. For this to work properly, you'll need to specify the slurm submission parameters for your institution using a `.yaml` file. The path to this file should be included in your configuration file under the key `slurm_config`. 

An example slurm submission:

```
dannce train dannce dannce_config.yaml --sbatch
```

By default, `predict-multi-gpu` will be submitted to the cluster.

An example slurm configuration file :

```
# Dannce slurm configuration
dannce_train: "--job-name=trainDannce -p olveczkygpu,gpu --mem=80000 -t 3-00:00 --gres=gpu:1 -N 1 -n 8"
dannce_train_grid: "--job-name=trainDannce -p olveczkygpu,gpu --mem=80000 -t 3-00:00 --gres=gpu:1 -N 1 -n 8"
dannce_predict: "--job-name=predictDannce -p olveczkygpu,gpu,cox,gpu_requeue --mem=30000 -t 1-00:00 --gres=gpu:1 -N 1 -n 8"
dannce_multi_predict: "--job-name=predictDannce -p olveczkygpu,gpu,cox,gpu_requeue --mem=30000 -t 0-03:00 --gres=gpu:1 -N 1 -n 8"

# Com slurm configuration
com_train: "--job-name=trainCom -p olveczkygpu,gpu --mem=30000 -t 3-00:00 --gres=gpu:1 -N 1 -n 8"
com_predict: "--job-name=predictCom -p olveczkygpu,gpu,cox,gpu_mig,gpu_requeue --mem=10000 -t 1-00:00 --gres=gpu:1 -N 1 -n 8"
com_multi_predict: "--job-name=predictCom -p olveczkygpu,gpu,cox,gpu_requeue --mem=10000 -t 0-03:00 --gres=gpu:1 -N 1 -n 8"

# Inference
inference: '--job-name=inference -p olveczky,shared --mem=30000 -t 3-00:00 -N 1 -n 8 --constraint="intel&avx2"'
# Setup functions (optional, set to "" if no setup is required. Trailing ; is required)
setup: "module load Anaconda3/2020.11; module load cuda/11.1.0-fasrc01; module load cudnn/8.0.4.30_cuda11.1-fasrc01; source activate sdannce;"
```
