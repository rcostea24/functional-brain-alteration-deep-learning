# Functional Brain Alteration Deep Learning

This repository contains a PyTorch-based deep learning framework for predicting cognitive states (such as Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN)) from functional/structural brain scans (e.g., fMRI) in NIfTI format.

## Repository Structure

- `main.py`: The entry point for running training or evaluation experiments.
- `dataloading/`: Contains `dataset.py` with a custom PyTorch `Dataset` (`NiftiDataset`) for loading and preprocessing `.nii` files. It integrates `monai` for spatial transforms and supports data oversampling.
- `models/`: Contains implementations of various deep learning architectures tailored for spatial-temporal data:
  - 3D Convolutional Networks (e.g., `ResNet3D`, `FullConvNet`)
  - Recurrent Neural Networks (`GRU`)
  - Transformer-based models (`Transformer`)
  - Spatial-Temporal combined models (`SpatialTemporalModel`)
  - Septr components (`SeptrConvNet`, `SeptrModel`, etc.)
- `training/`: Contains `trainer.py` to handle the training loop, model optimization, and `logger.py` for logging experiment parameters and metrics.
- `evaluation/`: Contains `evaluator.py` for evaluating trained models on test sets and computing metrics like Accuracy, Precision, Recall, and F1-Score.
- `experiments/`: Stores JSON configuration files (`exp_*.json`) that define all hyperparameters and settings for individual experiments.
- `experiments_done/`: Directory used to store completed experiment configurations.

## Setup and Requirements

The dependencies for this project are listed in `requirement.txt`. To install them, run:

```bash
pip install -r requirement.txt
```

Key libraries used in this project include:
- `torch` and `torchvision` for deep learning
- `nibabel` for loading NIfTI files
- `monai` for medical image augmentations/transforms
- `pandas` and `numpy` for data manipulation
- `opencv-python` (cv2) and `matplotlib` for visualization
- `tqdm` for progress tracking

## Usage

The project is designed to be highly configurable via JSON experiment files. An example configuration looks like this:

```json
{
    "exp_id": "051",
    "device": "cuda",
    "data_path": "data/adni_preprocessed",
    "model_name": "SpatialTemporalModel",
    "num_classes": 3,
    "batch_size": 32,
    "epochs": 150,
    ...
}
```

### Training

To train a model, you can run `main.py` with the `--mode train` argument. You can either specify a specific experiment ID or let it iterate over all configurations in the `experiments/` directory.

```bash
# Run a specific experiment (e.g., exp_051.json)
python main.py --mode train --exp_id 051 --data_root_path /path/to/data/root

# Run all experiments in the experiments/ folder
python main.py --mode train --data_root_path /path/to/data/root
```

### Evaluation

To evaluate a trained model, set the `--mode` argument to `evaluate`. Make sure your configurations are properly placed in the `experiments_eval` directory (or modify the root inside `main.py`).

```bash
# Evaluate a specific experiment
python main.py --mode evaluate --exp_id 051 --data_root_path /path/to/data/root
```

## Data Format

The dataloader expects the data to be structured inside the `--data_root_path` as specified in the experiment's JSON config (e.g., `data_path` parameter).
- It expects CSV annotation files named `{split}_annot.csv` (e.g., `train_annot.csv`, `val_annot.csv`).
- NIfTI image files should be stored in `{split}_img` directories (e.g., `train_img`, `val_img`).
- Labels can be dynamically mapped or excluded via the `"labels_mapping"` parameter in the configuration file.

## Logging

During training, metrics and experiment configurations are logged inside the `logs/` directory as text files (e.g., `log_train_051.txt`). Model checkpoints and other outputs will be saved according to the logic defined in `trainer.py`.