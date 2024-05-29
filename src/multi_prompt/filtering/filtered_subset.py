###
# Searching indexes of nonsense responses to
# form a subset without effect of safety module
# Index is based on lingual pos which means it's a relative index to lingual pos index of origin conversation

# For hybrid.json, zero-shot, (icl)

###


import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os
import re

from Testing.multi_prompt.pipeline.run import det_cor_gen, gpt_call4, lingual_detect_prompt

def filtering_none(sen):
    flag = True
    nonsense_set = ['sorry', 'Sorry', '抱歉']

    for word in nonsense_set:
        if word in sen:
            flag = False

    # lines = [l for l in sen.splitlines() if l.strip()]
    # if len(lines) < 12 or ('P' not in sen):
    #     flag = False

    return flag

def bcot_filtering_none(sen):
    lines = [l for l in sen.splitlines() if l.strip()]
    if len(lines) < 12 or ('P' not in sen):
        return False
    else:
        return True


def bcot_filtering_ori(sen):
    sen_split = re.split('[:：]', sen)
    if len(sen_split) != 0:
        return sen_split[-1]

    else:
        lines = [l for l in sen.splitlines() if l.strip()]
        return lines[-1]


def nonsense_idx(coach, file=None):
    none_idx = {}
    none_num = 0
    none_coach = {}
    if file == 'bcot':
        filtering_func = bcot_filtering_none
    else:
        filtering_func = filtering_none


    for key, lingual_idx_list in lingual_pos.items():
        none_idx[key] = []
        none_coach[key] = []
        for lingual_idx in lingual_idx_list:
            sen = coach[key][lingual_idx]
            flag = filtering_func(sen)
            if not flag:
                none_idx[key].append(lingual_idx)
                none_num += 1
                none_coach[key].append(sen)

    print(none_num)
    return none_idx

def merge_idx(dict1, dict2):
    merged_dict = {}

    for key in set(dict1.keys()).union(dict2.keys()):
        if key in dict1 and key in dict2:
            # Merge lists if key is in both dictionaries
            merged_dict[key] = list(set(dict1[key] + dict2[key]))
        elif key in dict1:
            # Add from dict1
            merged_dict[key] = dict1[key]
        else:
            # Add from dict2
            merged_dict[key] = dict2[key]

    return merged_dict


if __name__ == '__main__':

    with open('../../human_label_data/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)
    lingual_pos = {int(key): value for key, value in lingual_pos.items()}


    file = 'important cases/hybrid_generated.npy'
    coach = np.load(file, allow_pickle=True).item()
    none_idx1 = nonsense_idx(coach, file)

    file = 'important cases/zero_generated.npy'
    coach = np.load(file, allow_pickle=True).item()
    none_idx2 = nonsense_idx(coach)

    file = 'old/bcot_recap_generated.npy'
    coach = np.load(file, allow_pickle=True).item()
    none_idx3 = nonsense_idx(coach, file='bcot')

    merge_idx_dict = merge_idx(none_idx1, none_idx2)
    merge_idx_dict = merge_idx(merge_idx_dict, none_idx3)

    re_filtered_lingual_pos = {}
    re_filtered_lingual_pos_relative = {}
    for key, idx_list in lingual_pos.items():
        re_filtered_lingual_pos_relative[key] = []
        re_filtered_lingual_pos[key] = []
        for i, idx in enumerate(idx_list):
            if idx not in none_idx3[key]:
                re_filtered_lingual_pos_relative[key].append(i)
                re_filtered_lingual_pos[key].append(idx)


    with open('../../human_label_data/recap_filtered_lingual_pos', 'wb') as f:
        pickle.dump(re_filtered_lingual_pos, f)

    with open('../../human_label_data/recap_filtered_lingual_pos_relative', 'wb') as f:
        pickle.dump(re_filtered_lingual_pos_relative, f)












