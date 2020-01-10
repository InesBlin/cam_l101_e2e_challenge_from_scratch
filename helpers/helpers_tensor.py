# -*- coding: utf-8 -*-
import tensorflow as tf

def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def loss_coverage(att_weights, cov_vector):
    att_weights = tf.squeeze(att_weights).numpy()
    cov_vector = tf.squeeze(cov_vector).numpy()

    loss = 0
    nb_l = att_weights.shape[0]
    for i in range(nb_l):
        curr_att_w = att_weights[i, :]
        curr_cov_vector = att_weights[i, :]

        for index, val in enumerate(curr_att_w):
            if val <= curr_cov_vector[index]:
                loss += val
            else:
                loss += curr_cov_vector[index]
    
    return loss