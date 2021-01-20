import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):

    qk = tf.matmul(query, key, transpose_b= True)

    depth = tf.cast(tf.shape(key)[-1],tf.float32)
    scaled = qk / tf.math.sqrt(depth)

    if mask is not None:
        scaled += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

