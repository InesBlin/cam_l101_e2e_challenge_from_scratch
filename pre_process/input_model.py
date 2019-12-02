# -*- coding: utf-8 -*-
from helpers.helpers_sent import preprocess_sentence

class InputModelTrain:

    def __init__(self, row, mention_repr='seq'):
        self.row = row
        self.mention_repr = mention_repr
        self.nl_pre_processed = None
        self.input_encoder = []
    
    def transform_mention_repr(self):
        slot_value_list = self.row['mr'].split(', ')

        if self.mention_repr == 'seq': 
            # Input to encoder as [slot_1, value_1, ..., slot_n, value_n]
            for slot_value in slot_value_list:
                begin_val, end_val = slot_value.find('['), slot_value.find(']')
                slot, value = slot_value[:begin_val], slot_value[begin_val+1:end_val]
                self.input_encoder += [slot, value]
        return self
    
    def pre_process_nl(self):
        self.nl_pre_processed = preprocess_sentence(self.row['ref'])
        return self
    
    def pre_process(self):
        self.transform_mention_repr()
        self.pre_process_nl()
        return self