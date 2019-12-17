# -*- coding: utf-8 -*-
import argparse
import pandas as pd

def convert_for_metrics(mr_ref_path, save_path):
    """ Converting to a .txt format compatible to use the given metrics
    See https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs 
    One reference per line, different instances are separated by empty lines. 
    mr_ref_path must be .csv file with 'mr' and 'ref' columns
    save_path must be text file """
    df = pd.read_csv(mr_ref_path, sep=',', encoding='UTF-8')
    last_mr = None
    f = open(save_path, "w+")

    for _, row in df.iterrows():
        curr_mr, ref = row['mr'], row['ref']
        if last_mr!= curr_mr:  # New instance, adding empty line and updating last mr
            f.write('\n')
            last_mr = curr_mr
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
    # csv ./e2e-dataset/init-data/devset.csv
    # txt ./e2e-dataset/compat-eval-metrics/devset-compat.txt


