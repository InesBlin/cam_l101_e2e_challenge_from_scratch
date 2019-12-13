# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
from helpers.helpers_tensor import tokenize
from pre_process.input_model import InputModelTrain

def create_list(path_to_file, num_example):
    df = pd.read_csv(path_to_file, sep=',', encoding='UTF-8')
    mr, nl = [], []
    num_example = float('inf') if num_example is None else num_example

    for index, row in df.iterrows():
        if index < num_example:
            curr_da = InputModelTrain(row)
            curr_da.pre_process()
            mr.append(curr_da.input_encoder)
            nl.append(curr_da.nl_pre_processed)
    return mr, nl


def create_list_mr(path_to_file, num_example):
    df = pd.read_csv(path_to_file, sep=',', encoding='UTF-8')
    mr_raw, mr_input = [], []
    num_example = float('inf') if num_example is None else num_example

    for index, row in df.iterrows():
        if index < num_example:
            curr_da = InputModelTrain(row)
            curr_da.pre_process()
            mr_raw.append(curr_da.input_raw)
            mr_input.append(curr_da.input_encoder)
    return mr_raw, mr_input


def load(path, num_examples=None):
    # creating cleaned input, output pairs
    mr, nl = create_list(path, num_examples)

    mr_tensor, mr_lang_tokenizer = tokenize(mr)
    nl_tensor, nl_lang_tokenizer = tokenize(nl)

    return mr_tensor, nl_tensor, mr_lang_tokenizer, nl_lang_tokenizer


def create_tensor(mr_tensor, nl_tensor, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((mr_tensor, nl_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
