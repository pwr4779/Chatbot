from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from attention import scaled_dot_product_attention
class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        # depth: 64
        self.depth = d_model // self.num_heads
        #WQ, WK, WV, WO
        self.query_dense(Dense(d_model))
        self.key_dense(Dense(d_model))
        self.value_dense(Dense(d_model))
        self.dense = Dense(d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshpe(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)

        attention, _ = scaled_dot_product_attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attnetion = tf.reshape(attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attnetion)

        return outputs

    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]