"""Data helpers shared by ensemble training and prediction."""

import os

import numpy as np


def get_other_dataset(directory, block_num, classes=("AD", "HC")):
    """Load one spatial block for all samples and one-hot encode AD/HC labels."""

    data = []
    labels = []
    block_name = f"block{block_num}"
    for category in classes:
        category_dir = os.path.join(directory, category, block_name)
        for filename in sorted(os.listdir(category_dir)):
            data.append(np.load(os.path.join(category_dir, filename)))
            labels.append(0 if category == "AD" else 1)

    data = np.expand_dims(np.asarray(data), axis=4)
    labels = np.eye(2)[np.asarray(labels)]
    return data, labels
