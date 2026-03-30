# Roofs 2026

Binary semantic segmentation pipeline for roof detection from RGB aerial tiles using a U-Net++ model, XLSX-based dataset indexing, distributed training, evaluation tracking, visualization exports, and HPC-ready execution scripts.

## Overview

The project is organized around a single entrypoint, `train.py`, which supports:

- dataset preprocessing into an XLSX train/test index
- model training
- checkpoint-based test evaluation
- single-GPU and multi-GPU execution with DDP
- side-by-side visualization exports for augmentations and predictions

The current model is U-Net++ with:

- input shape: `3 x 1000 x 1000`
- binary mask shape: `1 x 1000 x 1000`
- optimizer: `AdamW`
- default loss: recall-oriented `TverskyFocalLoss`
- scheduler: `PolyLR`

## Repository Layout

```text
roofs_2026/
├── augmentations/
├── dataset/
├── evaluation/
├── hpc/
├── losses/
├── models/
├── preprocessing/
├── schedulers/
├── utils/
├── visualization/
├── logs/
├── runs/
├── requirements.txt
└── train.py
```

Key directories:

- `models/`: model definitions
- `preprocessing/`: XLSX train/test dataset generation
- `dataset/`: PyTorch dataset loading
- `augmentations/`: Albumentations pipelines
- `losses/`: segmentation loss functions
- `schedulers/`: learning-rate schedulers
- `evaluation/`: precision, recall, F1, IoU utilities
- `visualization/`: augmentation previews and prediction overlays
- `hpc/`: Apptainer and PBS job scripts
- `logs/`: PBS job logs in `YYYYMMDD-HHMMSS.log` format
- `runs/training/`: run outputs, metrics, configs, checkpoints, and visualizations

## Data Format

### Image Naming

Input images must follow:

```text
basename.tif
```

Example:

```text
5-1-1-102-11_x0_y0.tif
```

### Mask Naming

Input masks must follow:

```text
basename_mask.tif
```

Example:

```text
5-1-1-102-11_x0_y0_mask.tif
```

### Pair Matching Rule

Pairs are created by matching:

- image stem: `basename`
- mask stem: `basename_mask`

Only valid intersections are written to the XLSX dataset:

- masks without a matching image are skipped
- images without a matching mask are skipped

## Train/Test Dataset XLSX

The preprocessing stage creates an XLSX file with one row per valid image/mask pair.

Default path:

```text
dataset/train_test_dataset.xlsx
```

Columns:

- `tile_id`
- `image_path`
- `mask_path`
- `train_test`
- `fg_bg_ratio`
- `n_objects`
- `fg_area`

Column details:

- `train_test`: `train` or `test`
- `fg_bg_ratio`: foreground-to-background pixel ratio
- `n_objects`: number of connected foreground components in the mask
- `fg_area`: total foreground area in square meters

Foreground area is computed as:

```text
fg_area = fg_pixels * pixel_size_m^2
```

With the default `pixel_size_m=0.5`, one pixel corresponds to:

```text
0.25 m²
```

## Installation

### Python Environment

```bash
pip install -r requirements.txt
```

### Main Dependencies

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `openpyxl`
- `tifffile`
- `scipy`
- `albumentations`
- `opencv-python-headless`
- `Pillow`
- `tqdm`
- `PyYAML`

## Main Entry Point

All workflows go through:

```bash
python train.py
```

Supported modes:

- `preprocess`
- `train`
- `test`

## Command-Line Arguments

Common arguments:

- `--image-path`
- `--mask-path`
- `--train-test-dataset-path`
- `--output-dir`
- `--seed`
- `--pixel-size-m`

Training arguments:

- `--run-id`
- `--resume`
- `--n-epochs`
- `--batch-size`
- `--lr`
- `--loss`
- `--loss-weights`
- `--scheduler`
- `--n-devices`
- `--num-workers`
- `--base-channels`
- `--image-size`
- `--train-limit`
- `--test-limit`
- `--threshold`
- `--n-train-viz`
- `--n-test-viz`
- `--viz-augs`
- `--poly-power`
- `--min-lr`

Evaluation arguments:

- `--checkpoint`

Dataset refresh option:

- `--refresh-train-test-dataset`

## Preprocessing

Run preprocessing once to create the XLSX dataset:

```bash
python train.py \
  --mode preprocess \
  --image-path /path/to/images \
  --mask-path /path/to/masks \
  --train-test-dataset-path /path/to/train_test_dataset.xlsx \
  --test-size 0.2 \
  --seed 42 \
  --pixel-size-m 0.5
```

Behavior:

- scans all `.tif` and `.tiff` files recursively
- matches image/mask pairs by naming convention
- computes mask statistics
- assigns `train` and `test` labels
- saves the XLSX dataset

## Training

### Single-GPU Example

```bash
python train.py \
  --mode train \
  --image-path /path/to/images \
  --mask-path /path/to/masks \
  --train-test-dataset-path /path/to/train_test_dataset.xlsx \
  --run-id train_1gpu \
  --n-epochs 100 \
  --batch-size 2 \
  --lr 1e-4 \
  --loss tversky_focal \
  --scheduler poly \
  --n-devices 1 \
  --num-workers 4 \
  --base-channels 24 \
  --n-train-viz 100 \
  --n-test-viz 5
```

### Multi-GPU DDP Example

```bash
python train.py \
  --mode train \
  --image-path /path/to/images \
  --mask-path /path/to/masks \
  --train-test-dataset-path /path/to/train_test_dataset.xlsx \
  --run-id train_4gpu \
  --n-epochs 150 \
  --batch-size 2 \
  --lr 1e-4 \
  --loss tversky_focal \
  --scheduler poly \
  --n-devices 4 \
  --num-workers 4 \
  --base-channels 24
```

Notes:

- `batch-size` is per process, not global
- global batch size is `batch_size * n_devices`
- DDP is enabled automatically when `n_devices > 1`

## Resume Training

Resume continues the original run directory derived from the checkpoint path.

`--run-id` must not be used together with `--resume`.

Example:

```bash
python train.py \
  --mode train \
  --resume /path/to/runs/training/train_4gpu/checkpoints/best_f1_042.pt \
  --image-path /path/to/images \
  --mask-path /path/to/masks \
  --train-test-dataset-path /path/to/train_test_dataset.xlsx \
  --n-epochs 150 \
  --batch-size 2 \
  --lr 1e-4 \
  --loss tversky_focal \
  --scheduler poly \
  --n-devices 4 \
  --num-workers 4
```

Resume restores:

- model weights
- full model object
- model configuration
- optimizer state
- scheduler state
- AMP scaler state
- best F1 value and epoch
- best IoU value and epoch
- `evaluation.xlsx` continuation

## Test Evaluation

Evaluate a saved checkpoint:

```bash
python train.py \
  --mode test \
  --image-path /path/to/images \
  --mask-path /path/to/masks \
  --train-test-dataset-path /path/to/train_test_dataset.xlsx \
  --checkpoint /path/to/checkpoints/best_f1_042.pt \
  --batch-size 1 \
  --n-devices 1
```

## Checkpoints

The `checkpoints/` directory always contains exactly two files:

- `best_f1_{epoch}.pt`
- `best_iou_{epoch}.pt`

Only the latest best file for each criterion is kept:

- when a new best F1 is reached, the older `best_f1_*.pt` is removed
- when a new best IoU is reached, the older `best_iou_*.pt` is removed

Each checkpoint stores:

- full model object
- `model_state`
- `model_config`
- optimizer state
- scheduler state
- AMP scaler state
- run configuration
- metrics
- best F1 value and epoch
- best IoU value and epoch
- paths to the current best checkpoint files

## Run Outputs

Each training run is stored under:

```text
runs/training/<run_id>/
```

Typical contents:

```text
runs/training/<run_id>/
├── checkpoints/
│   ├── best_f1_XXX.pt
│   └── best_iou_XXX.pt
├── visualize/
│   ├── augs/
│   ├── train/
│   │   ├── per_epoch/
│   │   └── per_tile/
│   └── test/
│       ├── per_epoch/
│       └── per_tile/
├── config.json
├── config.yaml
└── evaluation.xlsx
```

## Metrics

Metrics are computed for both train and test splits:

- precision
- recall
- F1 score
- IoU
- loss

At the end of each epoch, the log prints:

- epoch id
- train loss
- test loss
- train precision, recall, F1, IoU
- test precision, recall, F1, IoU
- epoch duration
- best test F1 and its epoch
- best test IoU and its epoch

## Evaluation Workbook

Each run stores:

```text
runs/training/<run_id>/evaluation.xlsx
```

Columns:

- `epoch`
- `epoch duration sec`
- `lr`
- `train loss`
- `train prec`
- `train rec`
- `train f1`
- `train iou`
- `test loss`
- `test prec`
- `test rec`
- `test f1`
- `test iou`
- `best f1`
- `best f1 epoch`
- `best iou`
- `best iou epoch`

## Augmentations

Training augmentations include:

- random 90-degree rotations
- horizontal flip
- vertical flip
- transpose
- free-angle rotation
- shift, scale, and rotation
- random resized crop
- brightness and contrast adjustment
- hue, saturation, and value shift
- RGB shift
- Gaussian noise
- Gaussian blur
- motion blur

Evaluation transforms are empty by default.

## Visualizations

### Augmentation Visualizations

Saved to:

```text
runs/training/<run_id>/visualize/augs/
```

Each augmentation preview is a side-by-side image:

- left: augmented RGB image
- right: augmented binary mask
- separator: yellow vertical line

Mask interpretation:

- white: roof foreground
- black: background

### Prediction Visualizations

Saved to:

```text
runs/training/<run_id>/visualize/train/per_epoch/epoch_XXX/
runs/training/<run_id>/visualize/train/per_tile/<tile_id>/
runs/training/<run_id>/visualize/test/per_epoch/epoch_XXX/
runs/training/<run_id>/visualize/test/per_tile/<tile_id>/
```

Each prediction image is a side-by-side export:

- left: original RGB tile
- right: prediction overlay at threshold `0.5`
- separator: yellow vertical line

Overlay colors:

- green: true positive
- blue: false positive
- red: false negative

## Visualization Examples

Add final images after training to `docs/images/` and update the filenames below.

### Example 1: Train Split Prediction

Image file:

```text
docs/images/train_prediction_01.png
```

Description:

- left: original RGB tile from the train split
- right: prediction overlay at threshold `0.5`
- yellow line: panel separator
- green: true positive
- blue: false positive
- red: false negative

### Example 2: Train Split Prediction

Image file:

```text
docs/images/train_prediction_02.png
```

Description:

- left: original RGB tile from the train split
- right: prediction overlay at threshold `0.5`
- yellow line: panel separator
- green: true positive
- blue: false positive
- red: false negative

### Example 3: Test Split Prediction

Image file:

```text
docs/images/test_prediction_01.png
```

Description:

- left: original RGB tile from the test split
- right: prediction overlay at threshold `0.5`
- yellow line: panel separator
- green: true positive
- blue: false positive
- red: false negative

### Example 4: Augmentation Preview

Image file:

```text
docs/images/augmentation_01.png
```

Description:

- left: augmented RGB tile
- right: augmented binary mask
- yellow line: panel separator
- white mask pixels: roof foreground
- black mask pixels: background

## Logging

PBS jobs write logs to:

```text
logs/YYYYMMDD-HHMMSS.log
```

The timestamp is created when the job starts.

Convenience symlink:

```text
logs/latest.log
```

Follow the latest log:

```bash
tail -f logs/latest.log
```

## HPC Usage

### Preprocessing PBS Job

```bash
qsub hpc/submit_roofs_2026_preprocess.pbs
```

Current defaults:

- `1` CPU node
- `4` CPU cores
- `8 GB` RAM
- `8h` walltime

### Training PBS Job

```bash
qsub hpc/submit_roofs_2026.pbs
```

Current defaults:

- `1` node
- `4` GPUs
- `32` CPU cores
- `128 GB` RAM
- `96h` walltime
- `150` epochs
- per-device batch size `2`

Override selected variables at submit time:

```bash
qsub -v RUN_ID=train_4gpu_150ep_real,BATCH_SIZE=2,N_EPOCHS=150 hpc/submit_roofs_2026.pbs
```

Resume from a checkpoint through PBS:

```bash
qsub -v RESUME=/path/to/checkpoints/best_f1_042.pt hpc/submit_roofs_2026.pbs
```

## Container

Apptainer files:

- `hpc/roofs_2026.def`
- `hpc/roofs_2026.sif`
- `hpc/build_roofs_2026_sif.sh`

Build command:

```bash
bash hpc/build_roofs_2026_sif.sh
```

Direct container execution example:

```bash
apptainer exec --nv \
  --bind /path/to/roofs_2026:/workspace \
  /path/to/roofs_2026/hpc/roofs_2026.sif \
  python /workspace/train.py --mode train ...
```

## Current Dataset Paths

Current HPC dataset paths used in the PBS scripts:

```text
/lustre/group/IP-2022-10-2639/data/tiles_2022_2023
/lustre/group/IP-2022-10-2639/data/tiles_2022_2023_mask
```

## Notes

- `train.py` can regenerate the XLSX dataset with `--refresh-train-test-dataset`
- the PBS training script expects the XLSX dataset to already exist
- visualization exports are optional and controlled by `n_train_viz`, `n_test_viz`, and `viz_augs`
- non-interactive HPC logs print compact epoch summaries instead of verbose per-step progress lines
