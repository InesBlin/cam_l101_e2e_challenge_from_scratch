# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.settings import TRAIN_PATH_PRE_PROCESSED, MAIN_CONFIG_DELEX_PATH
from config.settings import DEV_PATH_PRE_PROCESSED_TEST, DEV_PATH_PRE_PROCESSED
from config.settings import GEN_SENT_PATH, SPECIFIC_FILE_NAME
from config.config import ConfigTrain

if __name__ == '__main__':

    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # config and train parameters permit to init the seq2seq model and restore the last checkpoint
    ap.add_argument("-config", "--config", help="main configuration path, containing main parameters " +
                                                "except the training dataset. See ./config/config_delex.yaml " +
                                                "for an example")
    ap.add_argument("-train", "--train", help="training data path")
    # two following arguments are path to the data + save path
    ap.add_argument("-test", "--test", help="path to file for generating sentences")
    ap.add_argument("-save", "--save", help="save path for generated sentences")

    args = vars(ap.parse_args())
    config = ConfigTrain(main_config_path=args['config'], train_path=args['train'])
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    seq2seq = Seq2SeqModel(config=config, 
                           optimizer=optimizer, loss_object=loss_object)
    seq2seq.restore_checkpoint()
    seq2seq.generate_sent(path_to_file=args['test'], save_path=args['save'])
    K.clear_session()
