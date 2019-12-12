# -*- coding: utf-8 -*-
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import Decoder
from helpers.helpers_tensor import loss_function
from pre_process.create_dataset import create_tensor


class Seq2SeqModel:

    def __init__(self, config, optimizer, loss_object):
        self.config = config
        self.optimizer = optimizer
        self.loss_object = loss_object

        self.dataset = create_tensor(mr_tensor=config.mr_tensor, nl_tensor=config.nl_tensor, 
                                     buffer_size=config.buffer_size, batch_size=config.batch_size)
        self.example_input_batch, _ = next(iter(self.dataset))

        # Write encoder and decoder model
        self.encoder = Encoder(config.vocab_mr_size, config.embedding_dim, 
                               config.units, config.batch_size)
        # sample input
        self.sample_hidden = self.encoder.initialize_hidden_state()
        self.sample_output, self.sample_hidden = self.encoder.call(self.example_input_batch, self.sample_hidden)

        self.attention_layer = BahdanauAttention(10)
        self.attention_result, self.attention_weights = self.attention_layer(self.sample_hidden, self.sample_output)

        self.decoder = Decoder(config.vocab_nl_size, config.embedding_dim, 
                               config.units, config.batch_size)
        self.sample_decoder_output, _, _ = self.decoder.call(tf.random.uniform((64, 1)), self.sample_hidden, self.sample_output)

        # Checkpoint
        self.checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    def print_info_shape(self):
        print ('Encoder output shape: (batch size, sequence length, units) {}'
                    .format(self.sample_output.shape))
        print ('Encoder Hidden state shape: (batch size, units) {}'
                    .format(self.sample_hidden.shape))
        print("Attention result shape: (batch size, units) {}"
                    .format(self.attention_result.shape))
        print("Attention weights shape: (batch_size, sequence_length, 1) {}"
                    .format(self.attention_weights.shape))
        print ('Decoder output shape: (batch_size, vocab size) {}'
                    .format(self.sample_decoder_output.shape))
    
    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder.call(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.config.nl_lang.word_index['<start>']] * self.config.batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder.call(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions, self.loss_object)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
    
    def train(self):
        for epoch in range(self.config.epochs):
            start = datetime.now()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.config.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                float(total_loss) / self.config.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))
    
    def evaluate(self, sentence):
        attention_plot = np.zeros((self.config.max_length_nl, self.config.max_length_mr))
        
        slot_value_list = sentence.split(', ')
        pre_inputs = []
        for slot_value in slot_value_list:
            begin_val, end_val = slot_value.find('['), slot_value.find(']')
            slot, value = slot_value[:begin_val], slot_value[begin_val+1:end_val]
            pre_inputs += [slot, value]

        pre_inputs = [elt.lower() for elt in pre_inputs]
        inputs_ = [self.config.mr_lang.word_index[elt] for elt in pre_inputs]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs_],
                                                                maxlen=self.config.max_length_mr,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, self.config.units))]
        enc_out, enc_hidden = self.encoder.call(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.config.nl_lang.word_index['<start>']], 0)

        for t in range(self.config.max_length_nl):
            predictions, dec_hidden, attention_weights = self.decoder.call(dec_input,
                                                                           dec_hidden,
                                                                           enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.config.nl_lang.index_word[predicted_id] + ' '

            if self.config.nl_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot
    
    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.config.checkpoint_dir))

    def translate(self, sentence):
        # self.checkpoint.restore(tf.train.latest_checkpoint(self.config.checkpoint_dir))
        result, sentence, _ = self.evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

                
