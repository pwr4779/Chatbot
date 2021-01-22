from flask import Flask, render_template, request
import re
import urllib.request
import pandas as pd
from Layer.transformer import transformer
import tensorflow as tf
import tensorflow_datasets as tfds
import os

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

path = './train/'
ckpt_1 = 'tf_chkpoint.ckpt'
model.load_weights(os.path.join(path, ckpt_1))

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)


  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    if tf.equal(predicted_id, END_TOKEN[0]):
      break
    output = tf.concat([output, predicted_id], axis=-1)
  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))
  return predicted_sentence


def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        question  = (request.form['question'])
        answer = model.predict(question)
        return render_template('index.html', answer=answer)
if __name__== '__main__':
    app.run(debug=True)

