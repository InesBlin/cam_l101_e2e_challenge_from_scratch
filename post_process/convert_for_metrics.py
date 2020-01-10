# -*- coding: utf-8 -*-
import argparse
import pandas as pd

def convert_ref(ref):
    # removing <end> if there is this pattern
    end_index = ref.find('<end>')
    if end_index != -1:
        ref = ref[:-(len(ref)-end_index)-1]
    
    # punctuations
    words = ref.split(' ')
    for index, word in enumerate(words):
        if (index <= len(words)-2) and (word == '.'):
            words[index+1] = words[index+1].capitalize()
    ref = ' '.join(words)

    ref = ref.replace(' .', '.')
    ref = ref.replace(' ,', ',')
    return ref

def convert_for_metrics(mr_ref_path, save_path, type_ref='generated'):
    """ Converting to a .txt format compatible to use the given metrics
    See https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs 
    One reference per line, different instances are separated by empty lines. 
    mr_ref_path must be .csv file with 'mr' and 'ref' columns
    save_path must be text file 
    if type_ref is 'init' => several instances for one MR
    if type_ref is 'generated' => one instance per line
    """
    df = pd.read_csv(mr_ref_path, sep=',', encoding='UTF-8')
    last_mr= None
    f = open(save_path, "w+", encoding='utf-8')

    if type_ref == 'init':
        for index, row in df.iterrows():
            curr_mr, ref = row['mr'], row['ref']
            if last_mr!= curr_mr:  # New instance, adding empty line and updating last mr
                if index != 0:
                    f.write('\n')
                last_mr = curr_mr
            ref = convert_ref(ref)
            f.write(ref + '\n')
        f.close()
    
    if type_ref == 'generated':
        for _, row in df.iterrows():
            curr_mr, ref = row['mr'], row['ref']
            if last_mr!= curr_mr:  # New instance, adding empty line and updating last mr
                last_mr = curr_mr
                ref = convert_ref(ref)
                f.write(ref + '\n')
        f.close()

if __name__=='__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-csv", "--csv", required=True, help="csv file to retrieve references")
    ap.add_argument("-txt", "--txt", required=True, help="txt save file")
    args = vars(ap.parse_args())

    convert_for_metrics(mr_ref_path=args['csv'], save_path=args['txt'])
    
    # For devset - Baseline example
    # csv ./e2e-dataset/generated-sent-lex/baseline.csv
    # txt ./e2e-dataset/compat-eval-metrics/devset-baseline.txt


