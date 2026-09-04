"""Spatial 3D-CNN base classifier described in the accompanying paper."""

import tensorflow as tf


class SpatialAttention(tf.keras.layers.Layer):
    """Channel pooling followed by a learnable 3D spatial attention map."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv3D(
            1, kernel_size=(3, 3, 3), padding="same", name="attention_conv"
        )
        self.activation = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        maximum = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = tf.concat([average, maximum], axis=-1)
        attention = self.activation(self.conv(attention))
        return inputs * attention


class Spatial3DCNN(tf.keras.Model):
    """One cube-level classifier in the 150-model Spatial 3D-CNN ensemble.

    Set ``return_features=True`` when extracting attended features for the
    meta-classifier and P-score pipeline.
    """

    def __init__(self, input_shape=(25, 25, 25, 1), return_features=False):
        super().__init__(name="spatial_3dcnn")
        self.return_features = return_features
        self.features = tf.keras.Sequential(
            [
                tf.keras.layers.Conv3D(
                    32, (4, 4, 4), input_shape=input_shape, name="conv3d_1"
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv3D(64, (3, 3, 3), name="conv3d_2"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
                tf.keras.layers.Conv3D(128, (3, 3, 3), name="conv3d_3"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv3D(128, (3, 3, 3), name="conv3d_4"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
            ],
            name="feature_extractor",
        )
        self.spatial_attention = SpatialAttention(name="spatial_attention")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.classifier = tf.keras.layers.Dense(2, activation="sigmoid")

    def call(self, inputs, training=None):
        local_features = self.features(inputs, training=training)
        attended_features = self.spatial_attention(local_features)
        output = self.classifier(self.dense1(self.flatten(attended_features)))
        if self.return_features:
            return output, attended_features
        return output
