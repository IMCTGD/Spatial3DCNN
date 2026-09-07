# Deep Learning Reveals Consistency and Complementarity of FDG-PET and Structural MRI in Alzheimer’s Pathological Progression

This repository contains the current Spatial Three-dimensional Convolutional Neural Network (Spatial 3DCNN) implementation and P-score pipeline used to compare FDG-PET and structural MRI in Alzheimer's pathological progression.

The model divides each preprocessed brain volume into 150 non-overlapping cubes of size `25 x 25 x 25`. A cube-level Spatial 3DCNN is trained independently for each position. The attended features of selected cube classifiers are then combined by a one-dimensional meta-classifier, whose learned spatial weights are used in the P-score pipeline. The same framework is used for both PET and structural MRI; the modality-specific data and result roots must be kept consistent.

## Model architecture

`spatial_3dcnn.py` implements the paper's cube-level classifier:

1. Conv3D(32, 4 x 4 x 4) + LayerNorm + ReLU
2. Conv3D(64, 3 x 3 x 3) + LayerNorm + ReLU
3. MaxPool3D(2 x 2 x 2)
4. Conv3D(128, 3 x 3 x 3) + LayerNorm + ReLU
5. Conv3D(128, 3 x 3 x 3) + LayerNorm + ReLU
6. Spatial attention using channel-wise average/max pooling and a 3 x 3 x 3 convolution
7. Flatten + Dense(128) + Dense(2)

`meta_classifier.py` implements the learnable spatial aggregation layer.

## Repository layout

```text
spatial3dcnn/
├── spatial_3dcnn.py          # Current Spatial 3DCNN base classifier
├── meta_classifier.py        # 1D spatial meta-classifier
├── train_spatial_3dcnn.py    # Train the 150 cube-level classifiers
├── evaluate_spatial_3dcnn.py # Validate/test cube-level classifiers
├── train_meta_classifier.py  # Train and evaluate the ensemble
├── prediction/               # Export sample-level predictions
├── dataset.py                # Dataset-list reader
├── data_utils.py             # Directory-based AD/HC loader
├── blockdataset.py           # Data generator and augmentation
├── ensemble_utils.py         # Feature aggregation and metrics
├── utils.py                  # Volume/block utilities
├── rank_cubes.py             # Rank cube-level validation results
├── analysis/                 # P-score, region, connectivity and progression analyses
│   ├── pscore_analysis.py    # Cube-to-region scores, mean/std and degeneration frequency
│   ├── connectivity_analysis.py # Spatial and spatiotemporal connected components
│   └── progression_analysis.py  # Longitudinal growth and MMSE trend statistics
└── dataprocessing/           # Dataset-list preparation
```

The legacy sMRI `PSN.py`/Ensemble 3DCNN implementation is intentionally not included.

## Environment

The paper experiments used TensorFlow 2.5.0. A compatible environment can be created with:

```bash
conda create -n spatial3dcnn python=3.8
conda activate spatial3dcnn
pip install -r requirements.txt
```

## Data organization

The scripts expect preprocessed and cube-split data following this pattern:

```text
<data-root>/
└── Fold/
    ├── fold1/block3D/
    │   ├── train/{AD,HC}/block0...block149/
    │   └── valid/{AD,HC}/block0...block149/
    └── test/block3D/{AD,HC}/block0...block149/
```

Each cube is stored as a NumPy array and receives a channel dimension when loaded. The PET volumes described in the paper were normalized and padded to `125 x 150 x 125` before being divided into 150 cubes; the corresponding sMRI preprocessing must produce the same cube layout.

## PET/sMRI modality switch

The current scripts expose the modality through the following variables:

| Purpose | PET | Structural MRI |
| --- | --- | --- |
| Data directory name | `PETData` | `ModalData` (or your MRI directory name) |
| Result directory name | `PETResult` | `MRIResult` |
| Data type label | `PET` | `MRI` |

These values are currently set inside `train_spatial_3dcnn.py`, `evaluate_spatial_3dcnn.py`, `train_meta_classifier.py`, `prediction/predict.py`, and `dataprocessing/createtxt.py`. Change all relevant values together before an experiment. In particular, `prediction/predict.py` contains a legacy hard-coded `PETData` test path that must be changed when predicting sMRI. Do not mix `MRIResult` checkpoints with PET data or vice versa.

## P-score and downstream analyses

The `analysis/` package is the cleaned, reusable version of the corresponding
legacy post-processing code. It contains no old `PSN` imports and no server
paths. It normalizes cube predictions, maps scores to Brainnetome regions by
voxel overlap, calculates region mean/std and `mean + 2*std` degeneration
frequency, constructs spatial/spatiotemporal connected components, and
summarizes continuous regional growth and normalized total P-score by MMSE.

These functions were refactored from the old `code/code/score.py`,
`score_utils_test.py`, `other_data/valid_oasis/valid_train_oasis_connect_utils.py`
and `analysis_train_valid.py`. The old scripts were not copied because they
depend on obsolete model classes and hard-coded `/home/lgy` paths.

## Workflow

1. Update the data/result roots, fold, and GPU selection in the script entry points.
2. Set the modality-specific data/result variables consistently, then run `python train_spatial_3dcnn.py` to train all cube-level Spatial 3DCNN classifiers.
3. Run `python evaluate_spatial_3dcnn.py` and `python rank_cubes.py` to evaluate and rank cubes.
4. Run `python train_meta_classifier.py` to train the spatial meta-classifier.
5. Run `python -m prediction.predict` to export predictions.

The repository contains no participant data, checkpoints, generated results, or legacy model code. Do not commit protected neuroimaging data or participant identifiers.
