"""Utilities for feature aggregation, evaluation, and P-score experiments."""

import sys

import numpy as np


def train_gen(train_data, train_label, target_augmented_samples=100):
    """Append randomly flipped samples used by the original ensemble pipeline."""

    augmented_data = []
    augmented_labels = []
    while len(augmented_data) < target_augmented_samples:
        index = np.random.randint(0, len(train_data))
        sample = train_data[index]
        changed = False
        for axis in range(3):
            if np.random.uniform(0, 1) > 0.5:
                sample = np.flip(sample, axis=axis)
                changed = True
        if changed:
            augmented_data.append(sample)
            augmented_labels.append(train_label[index])

    return (
        np.concatenate((train_data, np.asarray(augmented_data)), axis=0),
        np.concatenate((train_label, np.asarray(augmented_labels)), axis=0),
    )


def separate_samples(block_outputs):
    """Convert [block, sample, spatial features] to [sample, feature, block]."""

    outputs = np.asarray(block_outputs)
    samples = np.transpose(outputs, (1, 0, 2, 3, 4, 5))
    samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    return np.transpose(samples, (0, 2, 1))


def evaluate_feature(samples, labels, model, weight):
    """Evaluate the meta-classifier after assigning a set of cube weights."""

    layer = model.get_layer("spatial_weights")
    raw_weights = layer.get_weights()
    raw_weights[0] = np.reshape(weight, raw_weights[0].shape)
    layer.set_weights(raw_weights)
    loss, accuracy, auc = model.evaluate(samples, labels)
    predictions = model.predict(samples)
    return accuracy, loss, auc, predictions


def softmax(values):
    values = np.asarray(values) - np.max(values)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials)


def mcc_compute(labels, predictions):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / (denominator + sys.float_info.min)
