# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import tensorflow as tf
from helpers.helpers_tensor import max_length
from helpers.helpers_sent import preprocess_sentence
from pre_process.create_dataset import load
from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import Decoder
from settings import TRAIN_PATH_PRE_PROCESSED

# Try experimenting with the size of that dataset
num_examples = 5000
mr_tensor, nl_tensor, mr_lang, nl_lang = load(path=TRAIN_PATH_PRE_PROCESSED, num_examples=num_examples)
print(mr_lang.word_index)

# Calculate max_length of the target tensors
max_length_nl, max_length_mr = max_length(nl_tensor), max_length(mr_tensor)
print(max_length_nl, max_length_mr)

# Creating dataset
BUFFER_SIZE = len(mr_tensor)
BATCH_SIZE = 64
steps_per_epoch = len(mr_tensor)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_mr_size = len(mr_lang.word_index)+1
vocab_nl_size = len(nl_lang.word_index)+1
# Write encoder and decoder model
encoder = Encoder(vocab_mr_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_nl_size, embedding_dim, units, BATCH_SIZE)


def evaluate(sentence):
  attention_plot = np.zeros((max_length_nl, max_length_mr))

  # sentence = preprocess_sentence(sentence)
  
  slot_value_list = sentence.split(', ')
  pre_inputs = []
  for slot_value in slot_value_list:
    begin_val, end_val = slot_value.find('['), slot_value.find(']')
    slot, value = slot_value[:begin_val], slot_value[begin_val+1:end_val]
    pre_inputs += [slot, value]

  pre_inputs = [elt.lower() for elt in pre_inputs]

#   for w in pre_inputs:
#     w = re.sub(r"([?.!,¿])", r" \1 ", w)
#     w = re.sub(r'[" "]+', " ", w)
#     w = re.sub(r"[^a-zA-Z?.!,¿0-9£]+", " ", w)
#     w = w.rstrip().strip()
#     w = w.lower()
  
  print(pre_inputs)
  inputs = [mr_lang.word_index[elt] for elt in pre_inputs]


  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_mr,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)
  print(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([nl_lang.word_index['<start>']], 0)

  for t in range(max_length_nl):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    # attention_weights = tf.reshape(attention_weights, (-1, ))
    # attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()
    print(predicted_id, nl_lang.index_word[predicted_id])

    result += nl_lang.index_word[predicted_id] + ' '

    if nl_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  # plot_attention(attention_plot, sentence.split(' '), result.split(' '))


optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate('name[Taste of Cambridge], eatType[restaurant], priceRange[£20-25], customer rating[3 out of 5]')