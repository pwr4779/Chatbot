import tensorflow as tf
from multiheadattention import MultiHeadAttention, create_padding_mask
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, LayerNormalization, Dense
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

  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
      })

  # residual + normalization
  attention1 = LayerNormalization(1e-6)(attention1 + inputs)

  # MultiHeadAttention
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
          'mask': padding_mask # 패딩 마스크
      })

  attention2 = Dropout(rate=dropout)(attention2)
  attention2 = LayerNormalization(1e-6)(attention2 + attention1)

  outputs = Dense(units=dff, activation='relu')(attention2)
  outputs = Dense(units=d_model)(outputs)

  outputs = Dropout(rate=dropout)(outputs)
  outputs = LayerNormalization(1e-6)(outputs + attention2)

  return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)