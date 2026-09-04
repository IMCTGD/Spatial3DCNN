"""Meta-classifier used to aggregate spatial cube classifiers."""

import tensorflow as tf


class SpatialMetaClassifier(tf.keras.Model):
    """Learn one weight per selected cube and produce the final prediction."""

    def __init__(self, block_num, feature):
        super().__init__(name="spatial_meta_classifier")
        initializer = tf.keras.initializers.Constant(1.0 / block_num)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(feature, block_num))
        self.spatial_weights = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=initializer,
            name="spatial_weights",
        )
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.spatial_weights(x)
        return self.classifier(self.flatten(x))
