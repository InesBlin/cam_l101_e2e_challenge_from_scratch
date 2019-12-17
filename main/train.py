# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.settings import TRAIN_PATH_PRE_PROCESSED, MAIN_CONFIG_PATH
from config.settings import TRAIN_PATH_INIT
from config.config import ConfigTrain

if __name__ == '__main__':
    config = ConfigTrain(main_config_path=MAIN_CONFIG_PATH, train_path=TRAIN_PATH_PRE_PROCESSED)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    seq2seq = Seq2SeqModel(config=config, 
                           optimizer=optimizer, loss_object=loss_object)
    seq2seq.train()
    K.clear_session()