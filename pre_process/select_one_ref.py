# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from helpers.helpers_sent import preprocess_sentence

def update_vocab(ref, vocab):
    words = ref.split(' ')
    for word in words:
        vocab[word] += 1
    return vocab


def build_vocab(path_data):
    data = pd.read_csv(path_data, sep=',')
    vocab = defaultdict(lambda: 0)
    for _, row in data.iterrows():
        vocab = update_vocab(ref=preprocess_sentence(row['ref']), vocab=vocab)
    return vocab


def get_mr_to_ref(path_data):
    mr_to_ref = defaultdict(list)
    data = pd.read_csv(path_data, sep=',')

    for _, row in data.iterrows():
        mr, ref = row['mr'], row['ref']
        mr_to_ref[mr].append(ref)
    return mr_to_ref


def count_avg_words(ref, vocab):
    count = 0
    words = ref.split(' ')
    for word in words:
        count += vocab[word]
    return float(count)/len(words)


def get_one_ref(l_ref, vocab):
    max_score = count_avg_words(ref=l_ref[0], vocab=vocab)
    ref = l_ref[0]

    for sent in l_ref[1:]:
        curr_max_score = count_avg_words(ref=sent, vocab=vocab)
        if curr_max_score > max_score:
            max_score = curr_max_score
            ref = sent
    
    return ref


def get_new_train_set(path_data, path_save):
    vocab = build_vocab(path_data)
    mr_to_ref = get_mr_to_ref(path_data)
    new_train_set = {'mr': [], 'ref': []}
    for mr, l_ref in mr_to_ref.items():
        new_train_set['mr'].append(mr)
        new_train_set['ref'].append(get_one_ref(l_ref=l_ref, vocab=vocab))
    df = pd.DataFrame(new_train_set)
    df.to_csv(path_save, index=False, sep=',')


if __name__=='__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="path to data for vocab and references")
    ap.add_argument("-s", "--save", required=True, help='path to save new sentences')
    args = vars(ap.parse_args())

    get_new_train_set(path_data=args['data'], path_save=args['save'])
    
    # For devset - Like SHEFF2 example
    # data ./e2e-dataset/pre-processed-data/trainset-delex.csv
    # save ./e2e-dataset/pre-processed-data/trainset-delex-one-ref.csv


