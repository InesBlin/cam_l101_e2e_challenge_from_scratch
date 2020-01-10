# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from copy import deepcopy
from os import listdir

def delex_mr_verbatim(mr, verbatim_slots=['name', 'near']):
    mr_delex = []
    to_change_nl = {}
    slot_value_list = mr.split(', ')
    for slot_value in slot_value_list:
        slot, value = slot_value.split('[')
        if slot in verbatim_slots:
            to_change_nl[value[:-1]] = 'X-{0}'.format(slot)
            value = 'X-{0}]'.format(slot)
        mr_delex.append('{0}[{1}]'.format(slot, value[:-1]))
    return ', '.join(mr_delex), to_change_nl

def find_digit_letter_version(s, nb_to_letter, letter_to_nb):
    res = [s]
    if s[0] in nb_to_letter.keys():
        to_use = nb_to_letter
    else:
        to_use = letter_to_nb
    words = s.split(' ')
    changed = []
    for word in words:
        if word in to_use.keys():
            changed.append(to_use[word])
        else:
            changed.append(word)
    res.append(' '.join(changed))
    return res


def delex_customer_rating(cs_val, ref):
    """ More advanced delexicalisation for customer rating """
    delex_cs = None
    keep_search = True
    nb_to_letter = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five'}
    letter_to_nb = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}
    if ('out of 5' in cs_val) or ('out of five' in cs_val):
        type_val = 'out_of'
        digit, letter = cs_val[0], nb_to_letter[cs_val[0]]
    else:
        type_val = 'adj'
    
    if type_val == 'out of':
        digit_letter_versions = find_digit_letter_version(s=cs_val, nb_to_letter=nb_to_letter, letter_to_nb=letter_to_nb)
        search_template_full = ['customer rating of {0}', '{0} customer rating', '{0} rating', '{0} rated']
        search_template_star = ['{0}-star', '{0} star']
        search_pattern = []
        pass



def delex_nl(nl, to_change_nl):
    nl_delex = deepcopy(nl)
    for key, value in to_change_nl.items():
        nl_delex = nl_delex.replace(key, value)
    return nl_delex


def delex(row):
    """ Delexicalising name and near, replaced by X-name and X-near 
    Example of MR input 
    `name[The Cambridge Blue], eatType[pub], food[English], priceRange[cheap], near[Café Brazil]`
    Example of NL output
    "Close to Café Brazil, The Cambridge Blue pub serves delicious Tuscan Beef for the cheap price of £10.50. Delicious Pub food. """
    mr, nl = row['mr'], row['ref']
    mr_delex, to_change_nl = delex_mr_verbatim(mr)
    nl_delex = delex_nl(nl=nl, to_change_nl=to_change_nl)
    return mr_delex, nl_delex


def convert_to_delex(path_init, path_save):
    df = pd.read_csv(path_init, sep=',', encoding='UTF-8')
    mr_new, nl_new = [], []

    for _, row in df.iterrows():
        mr_delex, nl_delex = delex(row)
        mr_new.append(mr_delex)
        nl_new.append(nl_delex)
    
    new_df = pd.DataFrame({'mr': mr_new, 'ref': nl_new})
    new_df.to_csv(path_save, sep=',', index=False)
    return


if __name__=='__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to data to delexicalise")
    args = vars(ap.parse_args())

    path_init = args['input']
    file_name = path_init.split('/')[-1].split('.')[0]
    path_save = './e2e-dataset/pre-processed-data/{0}-delex.csv'.format(file_name)

    if '{0}.csv'.format(file_name) not in listdir('./e2e-dataset/pre-processed-data/'):
        convert_to_delex(path_init=path_init, path_save=path_save)

