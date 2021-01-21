from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from Layer.attention import scaled_dot_product_attention


class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert  self.d_model % self.num_heads == 0
        # depth: 64
        self.depth =  self.d_model // self.num_heads
        #WQ, WK, WV, WO
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)
        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(lambda x:tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth)))(inputs)
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(attention)
        concat_attention = tf.keras.layers.Lambda(lambda x: tf.reshape(x,(batch_size, -1, self.d_model)))(attention)
        outputs = self.dense(concat_attention)
        return outputs

    def get_config(self):
        config = super(MultiHeadAttention,self).get_config()
        config.update({
            'num_heads':self.num_heads,
            'd_model':self.d_model,
        })
        return config

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]