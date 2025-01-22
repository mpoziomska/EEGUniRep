# EEGUniRep


## Project Description

VICReg is an implementation of the Variance-Invariance-Covariance Regularization (VICReg) method for self-supervised learning. This approach optimizes a model by enforcing three regularization criteria:

1. **Variance**: Ensures the embeddings maintain sufficient diversity across dimensions.
2. **Invariance**: Encourages consistency between embeddings of positive pairs, regardless of transformations.
3. **Covariance**: Penalizes redundancy by minimizing the cross-correlation between dimensions of embeddings.

This implementation supports EEG-based tasks and integrates with libraries such as Neptune for experiment tracking. It also provides configurable preprocessing, transformations, and feature classification capabilities.

## Installation
```
git submodule init
git submodule update
curl -sSL https://install.python-poetry.org | python3 -
python3 venv .venv
source .venv/bin.activate
cd eegunirep
poetry install
```

## Usage
```
run_vic
```

## Parameters

### Command-Line Arguments (argparse)

- `--config_path`: Path to the configuration file (default: `base`).
- `--device`: Specify the device for training, e.g., `cuda` or `cpu` (default: `cuda`).
- `--loglevel`: Logging level for the application, e.g., `INFO` or `DEBUG` (default: `INFO`).

### Key Configuration Parameters (config.ini)

#### [general]
- `seed`: Random seed for experiment reproducibility.
- `workers`: Number of worker threads for data loading.
- `exp_id`: Unique identifier for the experiment.
- `runs_pth`: Path to store experimental results.

#### [training]
- `epochs`: Number of training epochs.
- `train_batch_size`: Batch size for training.
- `eval_batch_size`: Batch size for evaluation.
- `base_lr`: Base learning rate for the optimizer.

#### [transformations]
- `len_crop_s`: Length of cropped EEG segments in seconds.
- `num_crops`: Number of crops per EEG signal.
- `noise_mean`: Mean of Gaussian noise added for transformation.
- `noise_std`: Standard deviation of Gaussian noise.
- `normalize`: Boolean flag to apply normalization to the data.

#### [model]
- `sim_coeff`: Weight for the invariance term.
- `std_coeff`: Weight for the variance term.
- `cov_coeff`: Weight for the covariance term.
- `attention_mode`: Type of attention mechanism (e.g., `cbam`).
- `mlp_len_list`: List defining the hidden layer sizes in the MLP.

For a detailed example configuration, refer to the provided `base.toml` template.
