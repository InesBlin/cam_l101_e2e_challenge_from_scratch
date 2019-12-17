E2E NLG Challenge (2017) - Re taken for Cambridge L101 course in 2019
-----------------------

Natural Language Generation Challenge - Retaken in late 2019

Installation
----------------------

#### Settings 


Pre process the data
python pre_process/convert.py -i ./e2e-dataset/init-data/trainset.csv
python pre_process/convert.py -i ./e2e-dataset/init-data/testset.csv
python pre_process/convert.py -i ./e2e-dataset/init-data/devset.csv

Train a model
Set up the right configurations
python main/train.py

For reproducibility
Baseline 
- Basic Seq2Seq configuration, delex name and near, encoder & decoder GRU
- Params: batch_size: 20, embedding_dim: 50, units: 128, epochs: 10 [taken from the tgen baseline]
- Adam optimizer, cross-entropy loss function

Generate sentences from trained model
In the config.yaml file, ensure the `checkpoint_dir` is set to the right folder to retrieve the good model
python main/generate.py

Post-process
Replace X-near and X-name with their values
Transform .csv with mr and ref (whether it is the gold standards of the sentences generated from the model) to the format used by the metrics given