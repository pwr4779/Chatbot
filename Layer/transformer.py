
from tensorflow.keras.layers import Lambda
from Layer.multiheadattention import create_padding_mask
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, LayerNormalization, Dense, Embedding
from Layer.multiheadattention import MultiHeadAttention
from Layer.positionalencoding import PositionalEncoding

def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

  inputs = Input(shape=(None,), name="inputs")
  dec_inputs = Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더의 룩어헤드 마스크(첫번째 서브층)
  look_ahead_mask = Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 디코더의 패딩 마스크(두번째 서브층)
  dec_padding_mask = Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # encoder
  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  # decoder
  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # Dense
  outputs = Dense(units=vocab_size, name="outputs")(dec_outputs)

  return Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def encoder_layer(dff, d_model, num_heads, dropout, name="enoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")

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
    add_attention = tf.keras.layers.add([inputs, attention])
    attention = LayerNormalization(epsilon=1e-6)(add_attention)

    outputs = Dense(dff, activation='relu')(attention)
    outputs = Dense(d_model)(outputs)

    outputs = Dropout(dropout)(outputs)
    add_attention = tf.keras.layers.add([attention, outputs])
    outputs = LayerNormalization(epsilon=1e-6)(add_attention)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = Input(shape=(None,), name="inputs")

  padding_mask = Input(shape=(1, 1, None), name="padding_mask")

  embeddings = Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.keras.layers.Lambda(lambda x: tf.math.sqrt(tf.cast(x, tf.float32)))(d_model)
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = Dropout(rate=dropout)(embeddings)

  #encoder stack: paper -> 6
  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])

  return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = Input(shape=(None, d_model), name="inputs")
  enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
  # look_ahead_mask(first sub_layer)
  look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
  padding_mask = Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
      })

  # residual + normalization
  add_attention = tf.keras.layers.add([attention1, inputs])
  attention1 = LayerNormalization(epsilon=1e-6)(add_attention)

  # MultiHeadAttention
  attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs, # Q != K = V
          'mask': padding_mask
      })

  attention2 = Dropout(rate=dropout)(attention2)
  add_attention = tf.keras.layers.add([attention2, attention1])
  attention2 = LayerNormalization(epsilon=1e-6)(add_attention)

  outputs = Dense(units=dff, activation='relu')(attention2)
  outputs = Dense(units=d_model)(outputs)

  outputs = Dropout(rate=dropout)(outputs)
  add_attention = tf.keras.layers.add([outputs, attention2])
  outputs = LayerNormalization(epsilon=1e-6)(add_attention)

  return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout,name='decoder'):
  inputs = Input(shape=(None,), name='inputs')
  enc_outputs = Input(shape=(None, d_model), name='encoder_outputs')

  look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
  padding_mask = Input(shape=(1, 1, None), name='padding_mask')

  embeddings = Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.keras.layers.Lambda(lambda x: tf.math.sqrt(tf.cast(x, tf.float32)))(d_model)
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = Dropout(rate=dropout)(embeddings)

  # decoder layer stack
  for i in range(num_layers):
    outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)