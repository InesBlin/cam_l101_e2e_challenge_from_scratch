# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K 
from model.seq2seq import Seq2SeqModel
from config.settings import TRAIN_PATH_PRE_PROCESSED, MAIN_CONFIG_DELEX_PATH
from config.settings import DEV_PATH_PRE_PROCESSED_TEST, DEV_PATH_PRE_PROCESSED
from config.settings import GEN_SENT_PATH, SPECIFIC_FILE_NAME
from config.config import ConfigTrain

if __name__ == '__main__':
    config = ConfigTrain(main_config_path=MAIN_CONFIG_DELEX_PATH, train_path=TRAIN_PATH_PRE_PROCESSED)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    seq2seq = Seq2SeqModel(config=config, 
                           optimizer=optimizer, loss_object=loss_object)
    seq2seq.restore_checkpoint()
    seq2seq.generate('name[X-name], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[X-near]')
    # seq2seq.generate("name[X-name], eatType[pub], food[Fast food], customer rating[5 out of 5], area[riverside]")
    # seq2seq.generate("name[X-name], eatType[restaurant], priceRange[more than £30], area[riverside]")
    # seq2seq.generate("name[X-name], eatType[coffee shop], food[Chinese], customer rating[low], area[city centre], familyFriendly[no]")
    # seq2seq.generate_sent(path_to_file=DEV_PATH_PRE_PROCESSED, 
    #                       fold_to_save=GEN_SENT_PATH,
    #                       file_name=SPECIFIC_FILE_NAME['baseline'])

    K.clear_session()
