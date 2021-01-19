from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 50
def tokenize_and_filter(tokenizer, inputs, outputs,START_TOKEN,END_TOKEN):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

  tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

  return tokenized_inputs, tokenized_outputs