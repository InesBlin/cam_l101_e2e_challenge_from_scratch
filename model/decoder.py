# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from model.attention import BahdanauAttention

class DecoderBase(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, beam_size):
        super(DecoderBase, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.beam_size = beam_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class DecoderBeam(DecoderBase):
    """ Choosing a beam_size of 1 => greedy search """
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, beam_size):
        super().__init__(vocab_size, embedding_dim, dec_units, batch_sz, beam_size)
    
    class SentInfo():
        def __init__(self, att_plot, res, dec_input, dec_hidden, enc_out, score, ended):
            self.att_plot = att_plot
            self.res = res
            self.dec_input = dec_input
            self.dec_hidden = dec_hidden
            self.enc_out = enc_out
            self.score = score
            self.ended = ended
    
    def init_decode_path(self, config, dec_input, dec_hidden, enc_out):
        stored_sent = []
        att_plot=np.zeros((config.max_length_nl, config.max_length_mr))

        predictions, dec_hidden, attention_weights = self.call(dec_input, dec_hidden, enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        att_plot[0] = attention_weights.numpy()

        top_scores_, top_indices_ = tf.nn.top_k(predictions[0], k=self.beam_size, sorted=True, name=None)
        top_scores, top_indices = tf.math.log(top_scores_).numpy(), top_indices_.numpy()

        for i in range(self.beam_size):
            stored_sent.append(DecoderBeam.SentInfo(att_plot=att_plot,
                                                    res=config.nl_lang.index_word[top_indices[i]] + ' ',
                                                    dec_input=tf.expand_dims([top_indices[i]], 0),
                                                    dec_hidden=dec_hidden,
                                                    enc_out=enc_out,
                                                    score=top_scores[i],
                                                    ended=False))
        return stored_sent

    def decode_path(self, config, init_layers, sentence_raw):
        dec_input = init_layers['dec_input']
        dec_hidden = init_layers['dec_hidden']
        enc_out = init_layers['enc_out']

        finished_sent = []
        continue_search = True
        
        for step in range(config.max_length_nl):
            if step == 0:  # Initialization, taking beam_size best predictions
                stored_sent = self.init_decode_path(config, dec_input, dec_hidden, enc_out)
            else:  
                new_stored = [] 
                for curr_sent in stored_sent:
                    if continue_search:
                        if curr_sent.ended:
                            finished_sent.append(curr_sent)
                            if len(finished_sent) == self.beam_size:
                                continue_search = False
                        else:
                            curr_pred, curr_dec_hidden, curr_att_w = self.call(curr_sent.dec_input,
                                                                               curr_sent.dec_hidden,
                                                                               curr_sent.enc_out)  
                            curr_att_w = tf.reshape(curr_att_w, (-1, ))
                            curr_sent.att_plot[0] = curr_att_w.numpy()  

                            top_scores_, top_indices_ = tf.nn.top_k(curr_pred[0], k=self.beam_size, sorted=True, name=None)
                            top_scores, top_indices = tf.math.log(top_scores_).numpy(), top_indices_.numpy()

                            for i in range(self.beam_size):
                                is_ended = config.nl_lang.index_word[top_indices[i]] == '<end>'
                                new_stored.append(DecoderBeam.SentInfo(att_plot=curr_sent.att_plot,
                                                                       res=curr_sent.res + config.nl_lang.index_word[top_indices[i]] + ' ',
                                                                       dec_input=tf.expand_dims([top_indices[i]], 0),
                                                                       dec_hidden=curr_dec_hidden,
                                                                       enc_out=enc_out,
                                                                       score=curr_sent.score + top_scores[i],
                                                                       ended=is_ended))
                
                if len(finished_sent) < self.beam_size:
                    to_find = self.beam_size - len(finished_sent)
                    scores = [sent.score for sent in new_stored]
                    best_scores_indexes = sorted(range(len(scores)), key = lambda sub: scores[sub])[-to_find:]
                    stored_sent = [new_stored[i] for i in best_scores_indexes] 
        
        stored_sent = stored_sent + finished_sent
        return stored_sent

    
