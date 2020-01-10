# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.settings import TRAIN_PATH_PRE_PROCESSED, MAIN_CONFIG_TEST_PATH, TRAIN_PATH_PRE_PROCESSED_ONE_REF
from config.settings import TRAIN_PATH_INIT
from config.config import ConfigTrain

config = ConfigTrain(main_config_path=MAIN_CONFIG_TEST_PATH, train_path=TRAIN_PATH_PRE_PROCESSED_ONE_REF)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

seq2seq = Seq2SeqModel(config=config, 
                       optimizer=optimizer, loss_object=loss_object)
seq2seq.restore_checkpoint()
seq2seq.generate('name[X-name], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[X-near]')
seq2seq.generate("name[X-name], eatType[pub], food[Fast food], customer rating[5 out of 5], area[riverside]")
seq2seq.generate("name[X-name], eatType[restaurant], priceRange[more than £30], area[riverside]")
seq2seq.generate("name[X-name], eatType[coffee shop], food[Chinese], customer rating[low], area[city centre], familyFriendly[no]")
seq2seq.generate("name[X-name], area[riverside], familyFriendly[no]")
seq2seq.generate("name[X-name], area[riverside], familyFriendly[no], near[X-near]")
seq2seq.generate("name[X-name], eatType[coffee shop], food[Chinese], customer rating[3 out of 5], area[riverside], familyFriendly[yes]")
seq2seq.generate("name[X-name], eatType[coffee shop], food[Chinese], customer rating[average], area[city centre], familyFriendly[no]")
seq2seq.generate("name[X-name], eatType[coffee shop], food[English], customer rating[5 out of 5], area[city centre], familyFriendly[no]")
seq2seq.generate("name[X-name], area[city centre], near[X-near]")
K.clear_session()