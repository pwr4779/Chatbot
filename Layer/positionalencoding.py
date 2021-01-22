import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer

#d_model: output dim

class PositionalEncoding(Layer):
    def __init__(self, pos, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos = pos
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.pos, self.d_model)

    def angle(self, pos, i, d_model):
        return pos / tf.pow(1000, (2 * (i // 2) / d_model))

    def positional_encoding(self, pos, d_model):
        rads = self.angle(
            pos = tf.range(pos, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = tf.cast(d_model, dtype=tf.float32))

        # index: 2i -> sin function
        sines = tf.math.sin(rads[:, 0::2])

        # index: 2i+1 -> cosin function
        cosines = tf.math.cos(rads[:, 1::2])

        rads = np.zeros(rads.shape)
        rads[:, 0::2] = sines
        rads[:, 1::2] = cosines
        pos_encoding = tf.constant(rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs, **kwargs):
        return tf.keras.layers.Lambda(lambda inputs: inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :])(inputs)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'pos': self.pos,
            'd_model': self.d_model,
            'pos_encoding': self.pos_encoding,
        })
        return config
