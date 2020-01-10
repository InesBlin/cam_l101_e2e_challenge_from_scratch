# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.settings import CONFIG_MODEL_TRAIN
from config.config import ConfigTrain

if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="Type of model to train. " +
                    "baseline_delex for name and near delex," +
                    "copy_mechanism for pointer generator, test for test")
    args = vars(ap.parse_args())

    type_model = args['mode']
    config_model = CONFIG_MODEL_TRAIN[type_model]

    config = ConfigTrain(main_config_path=config_model['main_config_params'], train_path=config_model['data_path'])
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    seq2seq = Seq2SeqModel(config=config, 
                           optimizer=optimizer, loss_object=loss_object)
    seq2seq.train()
    K.clear_session()