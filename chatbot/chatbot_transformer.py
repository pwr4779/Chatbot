
import pandas as pd
import re
import urllib.request
import tensorflow_datasets as tfds
import tensorflow as tf
from util import tokenize_and_filter
from Layer.transformer import transformer

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv", filename="../Dataset/ChatBotData.csv")
train_data = pd.read_csv('../Dataset/ChatBotData.csv')

questions = []
answers = []

for sentenceQ, sentenceA in zip(train_data['Q'], train_data['A']):
    sentenceQ = re.sub(r"([?.!,])", r" \1 ", sentenceQ)
    sentenceQ = sentenceQ.strip()
    questions.append(sentenceQ)
    sentenceA = re.sub(r"([?.!,])", r" \1 ", sentenceA)
    sentenceA = sentenceA.strip()
    answers.append(sentenceA)


tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
# print(VOCAB_SIZE)
sample_string = questions[20]

questions, answers = tokenize_and_filter(tokenizer, questions, answers,START_TOKEN,END_TOKEN)

BATCH_SIZE = 64
BUFFER_SIZE = 20000
split = len(questions)-int(len(questions)/2)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions[:10],
        'dec_inputs': answers[:10, :-1] # 디코더의 입력
    },
    {
        'outputs': answers[:10, 1:]  # 시작토큰 제거
    },
)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions[split:split+10],
        'dec_inputs': answers[split:split+10, :-1] # 디코더의 입력
    },
    {
        'outputs': answers[split:split+10, 1:]  # 시작토큰 제거
    },
)).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

#lrate=d−0.5model×min(step_num^−0.5, step_num×warmup_steps^−1.5)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = tf.constant(d_model,dtype=tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step, **kwargs):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)
    return tf.math.multiply(tf.math.rsqrt(self.d_model), tf.math.minimum(arg1, arg2))

  def get_config(self):
      config = {
          'd_model': self.d_model,
          'warmup_steps': self.warmup_steps,
      }
      return config

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)
  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)
  return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

# Hyper-parameters
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
MAX_LENGTH = 50

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=loss_function, metrics = [accuracy])

# # 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # 다섯 번째 에포크마다 가중치를 저장하기 위한 콜백을 만듭니다
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)
# model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_dataset, validation_data = val_dataset, epochs = 1,)
tf.saved_model.save(model, 'model')
print(model.summary())


