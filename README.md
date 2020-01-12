E2E NLG Challenge (2017) - Re taken for Cambridge L101 course in 2019
-----------------------

Natural Language Generation Challenge retaken in late 2019 - beginning 2020. The description of the challenge can be found [here](http://www.macs.hw.ac.uk/InteractionLab/E2E/).

We reimplemented from scratch a sequence-to-sequence based approach using Tensorflow 2.0.
We used en encoder-decoder architecture with Bahdanau attention as the core architecture. We also added (can be optionnally tuned on) a coverage mechanism.


Installation
----------------------

#### Cloning the repository 

* Clone this repo to your computer by using `git clone https://github.com/Blinines/cam_l101_e2e_challenge_from_scratch.git` in the folder you want the repo to be in.
* Get into the folder using `cd cam_l101_e2e_challenge_from_scratch`.


#### Install the requirements

* Make sure you have a correct environment created for this project. You might want to use a virtual environment.  All experiments run with Python 3.7.4. In any case, activate your environment. For Linux the command is `source venv/bin/activate`, for Windows it is `venv\Scripts\activate`, if the name of your virtual environment is `venv`.
 
* Install the requirements using `pip install -r requirements.txt`. Additionnally install module `jupyter` to display in VSC `.ipynb` file.

* Install the setup file using `python setup.py install`

* NB : when you make change to files which are used as modules in other files, you will likely need to re run the `python setup.py install` command.

Usage
-----------------------

Several steps are necessary in order to prepare the data, train a model, generate sentences and post-process it.

##### Pre process the data
* Please note that in the repo the delexicalised data is already in the `./e2e-dataset/pre-processed-data`.
* To delexicalise a file `data.csv`, run `python pre_process/convert.py -i data.csv`. The new file will automatically be saved in the `./e2e-dataset/pre-processed-data` folder.
* In the context of the challenge we ran the three following lines.

```python
python pre_process/convert_delex.py -i ./e2e-dataset/init-data/trainset.csv
python pre_process/convert_delex.py -i ./e2e-dataset/init-data/testset_w_refs.csv
python pre_process/convert_delex.py -i ./e2e-dataset/init-data/devset.csv
```

##### Train a model
* First, set up the right configuration for the model you want to train. You can typically use the `config/config.yaml file`. In this file, you will find several parameters, certain speak for themselves, certaine need some clarifications :
    * `num_examples`: By default None, i.e. all the sentences of the training set will be trained. If you specify a number n, the model will only be trained with the first n sentences.
    * `checkpoint_dir`: saving checkpoint path 
    * `decoder_type`: Should be `beam_search`. Choosing a `beam_size` of 1 afterwards is equivalent to applying a greedy search when decoding.
    * `reranker_type`: Can be configured as `max` or `gazetteer_slug`. If set to `max`, will return the sentence with the best score. If set to `gazetteer_slug`, will update sentences' scores with an alignment score. 
    * `gazetteer_reranker`: Should be `./config/gazetteer.yaml`. Path to the gazetteer hashmap.
    * `pointer_generator`: Activating copy mechanism or not. => Currently not working, please set to False.
    * `coverage_mechanism`: Activating coverage mechanism or not. Boolean.
    * `reweight_cov_loss`: In case the coverage mechanism is activated, reweight of the coverage loss.


* In the context of the challenge we trained several models
Do not forget to change the path of the checkpoints
    * Name and near delexicalised - Bahdanau attention - all sentences trained
`python ./main/train.py -config ./config/config_delex.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex.csv`
    * Name and near delexicalised - Bahdanau attention - one sentence/MR trained
`python ./main/train.py -config ./config/config_delex.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex-one-ref.csv`
    * Name and near delexicalised - Bahdanau attention with coverage mechanism- all sentences trained
`python ./main/train.py -config ./config/config_delex_coverage.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex.csv`
    * Name and near delexicalised - Bahdanau attention - one sentence/MR trained
`python ./main/train.py -config ./config/config_delex_coverage.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex-one-ref.csv`

Each model will be trained with Adam optimizer and cross-entropy loss function.


##### Generate sentences from trained model
To generate sentence, the script will first reload the seq2seq model and restore the last checkpoint found.
Hence do not forget to set the right parameters for the model you trained, especially   the path of the checkpoints. 

* In the context of the challenge, here are the commands we executed (devset)
    * Name and near delexicalised - Bahdanau attention - all sentences trained
    `python ./main/generate.py -config ./config/config_delex.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex.csv -test ./e2e-dataset/pre-processed-data/devset-delex.csv -save ./e2e-dataset/generated-sent-delex/devset-delex-bahdanau-all.csv`
    * Name and near delexicalised - Bahdanau attention - one sentence/MR trained
    `python ./main/generate.py -config ./config/config_delex.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex-one-ref.csv -test ./e2e-dataset/pre-processed-data/devset-delex.csv -save ./e2e-dataset/generated-sent-delex/devset-delex-bahdanau-one-ref.csv`
    * Name and near delexicalised - Bahdanau attention with coverage mechanism - all sentences trained
    `python ./main/generate.py -config ./config/config_delex_coverage.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex.csv -test ./e2e-dataset/pre-processed-data/devset-delex.csv -save ./e2e-dataset/generated-sent-delex/devset-delex-coverage-all.csv`
    * Name and near delexicalised - Bahdanau attention with coverage mechanism - one sentence/MR trained
    `python ./main/generate.py -config ./config/config_delex_coverage.yaml -train ./e2e-dataset/pre-processed-data/trainset-delex-one-ref.csv -test ./e2e-dataset/pre-processed-data/devset-delex.csv -save ./e2e-dataset/generated-sent-delex/devset-delex-coverage-one-ref.csv`


##### Post-process
* First creating the lexicalised version of the delexicalised sentences created. (for our purpose only name and near delexicalised)

Example of command to run
```python
python ./post_process/convert_to_lex.py -or ./e2e-dataset/init-data/testset_w_refs.csv -de ./e2e-dataset/generated-sent-delex/testset-delex-bahdanau-all.csv -s ./e2e-dataset/generated-sent-lex/testset-delex-bahdanau-all.csv`
```


* Transform .csv file to compatible .txt file used for metrics.

Example of command to run
`python ./post_process/convert_for_metrics.py -csv ./e2e-dataset/generated-sent-lex/testset-delex-bahdanau-all.csv -txt ./e2e-dataset/compat-eval-metrics/testset-delex-bahdanau-all.txt`
Transform .csv with mr and ref (whether it is the gold standards of the sentences generated from the model) to the format used by the metrics given


##### Evaluating with automatic metrics
We used the metrics provided by the E2E NLG Challenge, from https://github.com/tuetschek/e2e-metrics.
As it was designed to run on Linux and we couldn't run it on Windows, we executed this part separately.

Describing project architecture
----------------------------------

* [config](./config)  : configuration files  
    * [config_copy_mechanism](./config/config_copy_mechanism.yaml) : default config when incorporating copy mechanism. Currently not working.
    * [config_delex](./config/config_delex.yaml) : default config for basic configuration. 
    * [config](./config/config.yaml) : to use for experimenting. 
    * [config.py](./config/config.py) : config class
    * [gazetteer.yaml](./config/gazetteer.yaml) : hashmap slot values to expressions for handcrafted gazetteer
    * [settings](./config/setting.py) : path to datasets

* [e2e-dataset](./e2e-dataset) 
    * [compat-eval-metrics](./e2e-dataset/compat-eval-metrics) : compatible final files to use the e2e-metrics provided by the challenge
    * [generated-sent-delex](./e2e-dataset/generated-sent-delex) : delexicalised generated sentences
    * [generated-sent-lex](./e2e-dataset/generated-sent-lex) : lexicalised generated sentences
    * [init-data](./e2e-dataset/init-data) : data provided by the challenge
    * [pre-processed-data](./e2e-dataset/[pre-processed-data) : name and near delexicalised


* [helpers](./helpers)  : 
    * [helpers_sent](./helpers/helpers_sent.py)  
    * [helpers_tensor](./helpers/helpers_tensor.py)   

* [main](./main) : 
    * [generate](./generate.py) 
    * [train](./train.py) 

* [model](./model) 
    * [attention](./model/attention.py) 
    * [decoder](./model/decoder.py) 
    * [encoder](./model/encoder.py) 
    * [reranker](./model/reranker.py)   
    * [seq2seq](./model/seq2seq.py) 

* [post_process](./post_process) : 
    * [convert_for_metrics](./post_process/convert_for_metrics.py) : converting lexicalised csv file with MR and ref to .txt compatible with evaluation metrics
    * [convert_to_lex](./post_process/convert_to_lex.py) : delex => lex

* [pre_process](./pre_process) : 
    * [convert_delex](./pre_process/convert_delex.py) : lex => delex (name and near)
    * [create_dataset](./pre_process/create_dataset) : dataset for training
    * [input_model](./pre_process/input_model.py) : Input class 
    * [select_one_ref](./pre_process/select_one_ref.py) : For each MR, selecting one ref only (the one whose words has the highest average frequency in the training set)

* [training_cp](./training_cp) : 
    * [training_checkpoints_delex_bahdanau_all](./training_cp/training_checkpoints_delex_bahdanau_all) : regular with all mr/ref couples
    * [training_checkpoints_delex_bahdanau_one_ref](./training_cp/training_checkpoints_delex_bahdanau_one_ref) : regular with one ref/mr
    * [training_checkpoints_delex_coverage_all](./training_checkpoints_delex_coverage_all) : regular with coverage mechanism and all mr/ref couples
    * [training_checkpoints_delex_coverage_one_ref](./training_cp/training_checkpoints_delex_coverage_one_ref) : regular with coverage mechanism and one ref/mr

* [bibliography](./bibliography.bib) : 
* [README](./README.md) : 
* [requirements](./requirements.txt) : 
* [setup](./setup.py)
* [visualisations](./visualisations.ipynb)

