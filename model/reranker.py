# -*- coding: utf-8 -*-
from collections import defaultdict

class AlignBase():
    """ Performs alignment between the input MR and a given reference 
    Main point is to retrieve the `slot_value_align`
    e.g. slot_value_align = {'name=x-name': [1, 0]}
    1 indicates that the slot-value was in the MR, 0 that it was not in the reference """
    def __init__(self, mr_list, ref, gazetteer):
        pre_process_mr = [elt.lower() for elt in mr_list]
        pre_process_mr = ['_'.join(elt.split(' ')) for elt in pre_process_mr]
        self.mr_list = pre_process_mr

        end_ref_index = ref.find('<end>')
        self.ref = ref if end_ref_index is -1 else ref[:end_ref_index-len(ref)]

        self.gazetteer = gazetteer
        self.keys_g = list(gazetteer.keys())
        self.slot_value_align = defaultdict(lambda: [0,0])
    
    def align_mr(self):
        for i in range(len(self.mr_list)//2):
            slot, value = self.mr_list[2*i], self.mr_list[2*i+1]
            if slot in self.keys_g:
                self.slot_value_align['{0}={1}'.format(slot, value)][0] = 1
        return self

    def align_ref(self):
        for slot in self.keys_g:
            if slot in ['name', 'near']:  # Special case, only one possible value
                is_in_ref = any(elt in self.ref for elt in self.gazetteer[slot])
                if is_in_ref:
                    self.slot_value_align['{0}=x-{0}'.format(slot)][1] = 1

            elif slot == 'familyfriendly':  # Special case, ensuring treating `no` case first
                is_not_ff = any(elt in self.ref for elt in self.gazetteer[slot][False])
                if is_not_ff:
                    self.slot_value_align['{0}={1}'.format(slot, 'no')][1] = 1
                else:
                    is_ff = any(elt in self.ref for elt in self.gazetteer[slot][True])
                    if is_ff:
                        self.slot_value_align['{0}={1}'.format(slot, 'yes')][1] = 1
            
            else:  # All other possible slots
                for value in self.gazetteer[slot]:
                    is_in_ref = any(elt in self.ref for elt in self.gazetteer[slot][value])
                    if is_in_ref:
                        self.slot_value_align['{0}={1}'.format(slot, value)][1] = 1
        return self

    def align_all(self):
        self.align_mr()
        self.align_ref()

    def get_align_score(self):
        keys_align = list(self.slot_value_align.keys())
        N, N_u, N_o = 0, 0, 0  # SLUG notations
        for key in keys_align:
            if (self.slot_value_align[key][0] == 1) and (self.slot_value_align[key][1] == 1):  # In the MR and the ref
                N += 1
            if (self.slot_value_align[key][0] == 1) and (self.slot_value_align[key][1] == 0):  # In the MR but not in the ref
                N += 1
                N_u += 1
            if (self.slot_value_align[key][0] == 0) and (self.slot_value_align[key][1] == 1):  # Not in the MR but in the ref
                N_o += 1

        return float(N/((N_u+1)*(N_o+1)))
    


class ReRankerBase():
    """ Can perform different reranker type
    if set to 'max' => will return max score of given entries
    if set to 'gazetteer_slug' => will adjust the scores following SLUG winning system method """
    def __init__(self, reranker_type, gazetteer):
        self.reranker_type = reranker_type
        self.gazetteer = gazetteer

    def get_max_score(self, stored_sent):
        scores = [sent.score for sent in stored_sent]
        final_index = sorted(range(len(stored_sent)), key = lambda sub: scores[sub])[-1:][0]
        result, attention_plot = stored_sent[final_index].res, stored_sent[final_index].att_plot
        return result, attention_plot
    
    def get_best(self, sentence_input_list, stored_sent):
        if self.reranker_type == 'gazetteer_slug':
            # Updating best sentences with alignment score
            for sent in stored_sent:
                curr_align = AlignBase(mr_list=sentence_input_list, ref=sent.res, gazetteer=self.gazetteer)
                curr_align.align_all()
                curr_align_score = curr_align.get_align_score()
                sent.score *= curr_align_score

        result, attention_plot = self.get_max_score(stored_sent)
        return result, attention_plot 
    