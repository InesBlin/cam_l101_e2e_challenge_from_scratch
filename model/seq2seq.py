# -*- coding: utf-8 -*-
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import DecoderBeam
from model.reranker import ReRankerBase
from helpers.helpers_tensor import loss_function, loss_coverage
from pre_process.create_dataset import create_tensor, create_list_mr


def pre_process_sent(sentence):
    slot_value_list = sentence.split(', ')
    pre_inputs = []
    for slot_value in slot_value_list:
        begin_val, end_val = slot_value.find('['), slot_value.find(']')
        slot, value = slot_value[:begin_val], slot_value[begin_val+1:end_val]
        pre_inputs += [slot, value]
    return pre_inputs


class Seq2SeqModel:

    def __init__(self, config, optimizer, loss_object):
        # General parameters
        self.config = config
        # mr_vocab_size = len(self.config.mr_lang.word_index)
        #self.config.mr_lang.word_index['UNK'] = mr_vocab_size + 1
        self.config.mr_lang.word_index['UNK'] = None
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.dataset = create_tensor(mr_tensor=config.mr_tensor, nl_tensor=config.nl_tensor, 
                                     buffer_size=config.buffer_size, batch_size=config.batch_size)

        # Encoder
        self.encoder = Encoder(config.vocab_mr_size, config.embedding_dim, 
                               config.units, config.batch_size)

        # Decoder
        self.decoder = DecoderBeam(config.vocab_nl_size, config.embedding_dim, 
                                   config.units, config.batch_size, config.beam_size, 
                                   config.nl_lang.word_index, config.nl_lang.index_word,
                                   config.pointer_generator)
        self.reranker = ReRankerBase(config.reranker_type, config.gazetteer_reranker)

        # Checkpoint
        self.checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
    

    # @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder.call(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.config.nl_lang.word_index['<start>']] * self.config.batch_size, 1)

            if self.config.coverage_mechanism:
                coverage_vector = tf.zeros((self.config.batch_size, self.config.max_length_mr, 1))
            else:
                coverage_vector = None

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
                predictions, dec_hidden, attention_weights, p_gen, coverage_vector = self.decoder.call(dec_input, dec_hidden, 
                                                                                                       enc_output, coverage_vector)
                if self.config.pointer_generator:
                    predictions = self.decoder.update_prob_ditrib_pointer(predictions, attention_weights, p_gen, inp)
                
                # Updating losses : general + coverage mechanism if any
                loss += loss_function(targ[:, t], predictions, self.loss_object)
                if self.config.coverage_mechanism:
                    loss +=  self.config.reweight_cov_loss * loss_coverage(attention_weights, coverage_vector)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        # for var in variables:
        #     print(var)
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
    
    def evaluate(self, sentence_input_list, sentence_raw):

        pre_inputs = [elt.lower() for elt in sentence_input_list]
        inputs_ = []
        for elt in pre_inputs:
            if elt in self.config.mr_lang.word_index:
                inputs_.append(self.config.mr_lang.word_index[elt])
            else:
                inputs_.append(self.config.mr_lang.word_index['UNK'])
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs_],
                                                                maxlen=self.config.max_length_mr,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        hidden = [tf.zeros((1, self.config.units))]
        enc_out, enc_hidden = self.encoder.call(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.config.nl_lang.word_index['<start>']], 0)

        stored_sent = self.decoder.decode_path(config=self.config, 
                                               init_layers={'dec_input': dec_input, 
                                                            'dec_hidden': dec_hidden, 
                                                            'enc_out': enc_out}, 
                                               sentence_raw=sentence_raw,
                                               sentence_input_list=sentence_input_list)

        finished_sent = [sent for sent in stored_sent if sent.ended]
        if finished_sent:
            result, attention_plot = self.reranker.get_best(sentence_input_list, finished_sent)
        else:
            result, attention_plot = self.reranker.get_best(sentence_input_list, stored_sent)
            
        return result, sentence_raw, attention_plot
    
    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.config.checkpoint_dir))

    def generate(self, sentence_raw):
        sent_inp_l = pre_process_sent(sentence_raw)
        result, sentence, _ = self.evaluate(sent_inp_l, sentence_raw)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        print('===')

        return result
    
    def generate_sent(self, path_to_file, fold_to_save, file_name):
        mr_raw, mr_input = create_list_mr(path_to_file=path_to_file, num_example=self.config.num_examples)
        res = {'mr':[], 'ref': []}

        for index, mr_raw in enumerate(mr_raw):
            gen_sent, _, _ = self.evaluate(sentence_input_list=mr_input[index], sentence_raw=mr_raw)
            res['mr'].append(mr_raw)
            res['ref'].append(gen_sent)
        
        df = pd.DataFrame(res)
        df.to_csv(fold_to_save+file_name, index=False, sep=',')
        




                
