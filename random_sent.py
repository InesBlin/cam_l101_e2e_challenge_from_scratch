# -*- coding: utf-8 -*-
import pandas as pd 

def get_line_indexes(mr_ref_path, type_ref):
    """ Getting structure for different MR 
    See https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs 
    One reference per line, different instances are separated by empty lines. 
    mr_ref_path must be .csv file with 'mr' and 'ref' columns
    if type_ref is 'init' => several instances for one MR
    if type_ref is 'generated' => one instance per line
    """
    df = pd.read_csv(mr_ref_path, sep=',', encoding='UTF-8')
    last_mr= None
    num_sent_to_index = []

    if type_ref == 'init':
        curr_index = 0
        for index, row in df.iterrows():
            curr_mr = row['mr']
            if last_mr!= curr_mr:  # New instance, adding empty line and updating last mr
                if index == 0:
                    curr_index += 1
                    num_sent_to_index.append(curr_index)
                else:
                    curr_index += 2
                    num_sent_to_index.append(curr_index)
                
                last_mr = curr_mr
            else:
                curr_index += 1
        return num_sent_to_index
    
    if type_ref == 'generated':
        curr_index = 1
        for index, row in df.iterrows():
            curr_mr = row['mr']
            if last_mr!= curr_mr:  # New instance, adding empty line and updating last mr
                curr_index += 1
                num_sent_to_index.append(curr_index)
                last_mr = curr_mr
            else:
                curr_index += 1
        return num_sent_to_index


gold_path = './e2e-dataset/init-data/testset_w_refs.csv'
sent_gold = get_line_indexes(gold_path, 'init')
sent_model = get_line_indexes(gold_path, 'generated')

sent_random = [14, 131, 133, 184, 185, 187, 214, 217, 220, 233, 273, 318, 358, 385, 391, 423, 449, 508, 542, 571]
for sent_num in sent_random:
    print('sentence num: {0}\t begin in ref: {1}\t begin in model: {2}'.format(sent_num, 
                                                                               sent_gold[sent_num-1],
                                                                               sent_model[sent_num-1]))
