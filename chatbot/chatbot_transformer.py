
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from util import tokenize_and_filter

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

sample_string = questions[20]

questions, answers = tokenize_and_filter(tokenizer, questions, answers,START_TOKEN,END_TOKEN)

BATCH_SIZE = 64
BUFFER_SIZE = 20000

AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1] # 디코더의 입력
    },
    {
        'outputs': answers[:, 1:]  # 시작토큰 제거
    },
)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefecth(AUTOTUNE)

