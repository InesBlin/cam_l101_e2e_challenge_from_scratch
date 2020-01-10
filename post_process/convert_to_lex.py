# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pre_process.input_model import InputModelTrain


def create_slot_value(slot_value_list):
    slot_value_res = {}
    for slot_value in slot_value_list:
        begin_val, end_val = slot_value.find('['), slot_value.find(']')
        slot, value = slot_value[:begin_val], slot_value[begin_val+1:end_val]
        slot_value_res[slot] = value
    return slot_value_res


def replace_delex(row_delex, slot_value):
    delex_sent = row_delex['ref'].values[0]
    words = delex_sent.split(' ')
    new_sent, nb_words = [], len(words)

    for index, word in enumerate(words):
        if (index!=nb_words-1) and (word=='x'):
            delex_val = words[index+1]
            if delex_val in slot_value.keys():
                words[index+1] = slot_value[delex_val]
        else:
            new_sent.append(word)
    
    return ' '.join(new_sent)


def convert_final(orig_mr_path, delex_sent_path, to_save_path):
    orig_mr_df = pd.read_csv(orig_mr_path, sep=',', encoding='UTF-8')
    delex_df = pd.read_csv(delex_sent_path, sep=',', encoding='UTF-8')
    res = {'mr':[], 'ref': []}

    for index, row_orig in orig_mr_df.iterrows():
        row_delex = delex_df.iloc[[index]]
        slot_value = create_slot_value(row_orig['mr'].split(', '))
        res['mr'].append(row_orig['mr'])
        res['ref'].append(replace_delex(row_delex, slot_value))
    
    df = pd.DataFrame(res)
    df.to_csv(to_save_path, index=False, sep=',')


if __name__=='__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-or", "--origin", required=True, help="path to initial lexicalised data")
    ap.add_argument("-de", "--delex", required=True, help="path to sentences (unlexicalised)")
    ap.add_argument("-s", "--save", required=True, help='path to save new sentences')
    args = vars(ap.parse_args())

    convert_final(orig_mr_path=args['origin'], delex_sent_path=args['delex'],
                  to_save_path=args['save'])
    
    # For devset - Baseline example
    # origin ./e2e-dataset/init-data/devset.csv
    # delex ./e2e-dataset/generated-sent-delex/baseline.csv
    # save ./e2e-dataset/generated-sent-lex/baseline.csv

        
        
        
        
        