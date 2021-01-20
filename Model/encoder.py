import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout, LayerNormalization, Dense, Embedding
from multiheadattention import MultiHeadAttention
from positionalencoding import PositionalEncoding
def encoder_layer(dff, d_model, num_heads, dropout, name="enoder_layer"):
    inputs = Input(shape=(None, d_model), model="inputs")

    # padding mask
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask' : padding_mask
    })

    # drop_out + normalization
    attention = Dropout(rate=dropout)(attention)
    # Residual connection
    attention = LayerNormalization(1e-6)(input + attention)

    outputs = Dense(dff, activation='relu')(attention)
    outputs = Dense(d_model)(outputs)

    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(1e-6)(attention + outputs)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = Input(shape=(None,), name="inputs")

  padding_mask = Input(shape=(1, 1, None), name="padding_mask")

  embeddings = Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = Dropout(rate=dropout)(embeddings)

  #encoder stack: paper -> 6
  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])

  return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)