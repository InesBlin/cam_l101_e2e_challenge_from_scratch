# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.config import ConfigTrain

if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", "--config", help="main configuration path, containing main parameters " +
                                                "except the training dataset. See ./config/config_delex.yaml " +
                                                "for an example")
    ap.add_argument("-train", "--train", help="training data path")
    args = vars(ap.parse_args())

    config_path = args['config']
    train_path = args['train']

    config = ConfigTrain(main_config_path=config_path, train_path=train_path)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    seq2seq = Seq2SeqModel(config=config, 
                           optimizer=optimizer, loss_object=loss_object)
    seq2seq.train()
    K.clear_session()