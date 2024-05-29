import openai
import pickle
import numpy as np
import csv
import os
import re
import glob
from evaluate import load
import jieba
from annotation.metrics import qa_f1_zh_score, qa_f1_score, normalize_zh_answer, normalize_answer
from multi_prompt.pipeline.run import det_cor_process, Accuracy, Exact_Match, f1_score,MT_metrics


# ------------------------------------------------------

# Evaluating code for generated coach sentences and processed human labels

# ------------------------------------------------------

def lingual_evaluation(det_cor_dict, filename):

    with open('../../../human_label_data/all_cases', 'rb') as f:
        all_cases = pickle.load(f)

    # loading lingual pos
    with open('../../../human_label_data/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    # loading relative pos
    with open('relative_pos', 'rb') as f:
        relative_pos = pickle.load(f)


    with open('../../../human_label_data/human_detection_all_old', 'rb') as f:
        old_human_d = pickle.load(f)

    with open('../../../human_label_data/human_correction_all_old', 'rb') as f:
        old_human_c = pickle.load(f)

    lingual_human_detection = []
    lingual_human_correction = []
    for k, v in lingual_pos.items():
        k = str(k)
        for i in v:
            lingual_human_detection.append(all_cases[k]['detection'][i])
            lingual_human_correction.append(all_cases[k]['correction'][i])


    ### old human label
    old_human_detection = []
    old_human_correction = []
    for k, v in lingual_pos.items():
        # k = str(k)
        for i in v:
            old_human_detection.append(old_human_d[k][i])
            old_human_correction.append(old_human_c[k][i])

    old_human = {'det': old_human_detection, 'cor': old_human_correction}
    with open('important_case/testing_result/old_human', 'wb') as f:
        pickle.dump(old_human, f)

    # # other methods
    special_set = ['bcot.npy', 'cot2.npy', 'gpt_coach.npy']
    if filename in special_set:
        new_det_cor = {}
        for k, v in relative_pos.items():
            if len(v) > 0:
                new_det_cor[k] = [det_cor_dict[int(k)][p] for p in relative_pos[str(k)]]

    else:
        # bcot only
        new_det_cor = det_cor_dict

    processed = det_cor_process(new_det_cor)
    print(len(processed['det']))

    det_none_num = 0
    cor_none_num = 0
    for d, c in zip(processed['det'], processed['cor']):
        if d == 'None':
            det_none_num += 1
        if c == 'None':
            cor_none_num += 1

    print(f'filename: {filename}')
    print(f'detection none case: {det_none_num}/300, correction none case: {cor_none_num}/300')


    det_EM = Exact_Match(processed['det'], lingual_human_detection)
    print('det EM: ', det_EM)
    # print(det_EM)

    cor_EM = Exact_Match(processed['cor'], lingual_human_correction)


    det_mt_results = MT_metrics(processed['det'], lingual_human_detection)
    cor_mt_results = MT_metrics(processed['cor'], lingual_human_detection)



if __name__ == '__main__':
    files = glob.glob('gpt_coach.npy')

    for f in files:
        print(f'filename {f}\n\n')
        lingual_dict = np.load(f, allow_pickle=True).item()
        lingual_evaluation(lingual_dict, f)











