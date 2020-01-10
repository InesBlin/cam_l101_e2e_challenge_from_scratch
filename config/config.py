# -*- coding: utf-8 -*-
import yaml
from pre_process.create_dataset import load
from helpers.helpers_tensor import max_length

# with open('./config.yaml') as file:
#     config_global = yaml.load(file, Loader=yaml.FullLoader)

# with open("./config.yaml", "w") as f:
#     yaml.dump(config_global, f)

class ConfigTrain:
    def __init__(self, main_config_path, train_path):
        with open(main_config_path) as file:
            main_config = yaml.load(file, Loader=yaml.FullLoader)

        if main_config['num_examples'] == 'None':
            self.num_examples = None
        else:
            self.num_examples = int(main_config['num_examples'])
        self.batch_size = int(main_config['batch_size'])
        self.embedding_dim = int(main_config['embedding_dim'])
        self.units = int(main_config['units'])
        self.epochs = int(main_config['epochs'])
        self.checkpoint_dir = main_config['checkpoint_dir']
        self.decoder_type = main_config['decoder_type']
        self.beam_size = main_config['beam_size']
        self.reranker_type = main_config['reranker_type']
        with open(main_config['gazetteer_reranker'], encoding='utf8') as file:
            self.gazetteer_reranker = yaml.load(file, Loader=yaml.FullLoader)
        self.pointer_generator = main_config['pointer_generator']
        self.coverage_mechanism = main_config['coverage_mechanism']
        self.reweight_cov_loss = main_config['reweight_cov_loss']

        self.mr_tensor, self.nl_tensor, self.mr_lang, self.nl_lang = load(path=train_path, 
                                                                          num_examples=self.num_examples)
        self.max_length_nl = max_length(self.nl_tensor)
        self.max_length_mr = max_length(self.mr_tensor)
        self.buffer_size = len(self.mr_tensor)
        self.steps_per_epoch = len(self.mr_tensor)//self.batch_size
        self.vocab_mr_size = len(self.mr_lang.word_index)+1
        self.vocab_nl_size = len(self.nl_lang.word_index)+1

