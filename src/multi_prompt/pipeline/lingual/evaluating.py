import openai
import pickle
import numpy as np
import csv
import os
import re
import glob
import json
from evaluate import load
import jieba
from annotation.metrics import qa_f1_zh_score, qa_f1_score, normalize_zh_answer, normalize_answer
from multi_prompt.pipeline.coach_run import det_cor_process, Accuracy, Exact_Match, f1_score,MT_metrics


# ------------------------------------------------------

# Evaluating generated coach sentences. Only lingual part

# ------------------------------------------------------

def human_label():
    # loading human label from all case dict
    with open('../../../compressed_dict/all_cases', 'rb') as f:
        all_cases = pickle.load(f)

    # loading lingual pos
    with open('../../../compressed_dict/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    # loading human label
    lingual_human_detection = []
    lingual_human_correction = []
    for k, v in lingual_pos.items():
        k = str(k)
        for i in v:
            lingual_human_detection.append(all_cases[k]['detection'][i])
            lingual_human_correction.append(all_cases[k]['correction'][i])

    return lingual_human_detection, lingual_human_correction


def lingual_evaluation(det_cor_dict, filename):
    # loading relative pos
    with open('relative_pos', 'rb') as f:
        relative_pos = pickle.load(f)

    # idx alignment
    new_det_cor = {}
    for k, v in relative_pos.items():
        if len(v) > 0:
            new_det_cor[k] = [det_cor_dict[int(k)][p] for p in relative_pos[str(k)]]

    processed = det_cor_process(new_det_cor)

    # # counting none cases
    # det_none_num = 0
    # cor_none_num = 0
    # for d, c in zip(processed['det'], processed['cor']):
    #     if d == 'None':
    #         det_none_num += 1
    #     if c == 'None':
    #         cor_none_num += 1
    #
    # print(f'filename: {filename}')
    # print(f'detection none case: {det_none_num}/300, correction none case: {cor_none_num}/300')

    human_detection, human_correction = human_label()
    det_em = Exact_Match(processed['det'], human_detection)
    cor_em = Exact_Match(processed['cor'], human_correction)

    det_mt_results = MT_metrics(processed['det'], human_detection)
    cor_mt_results = MT_metrics(processed['cor'], human_detection)

    eval_dict = {'det_em': det_em, 'cor_em': cor_em, 'det_mt': det_mt_results, 'cor_mt': cor_mt_results}
    print(eval_dict)

    with open('eval/eval_'+filename, 'wb') as f:
        json.dump(eval_dict, f)


if __name__ == '__main__':
    files = glob.glob('data/')

    for f in files:
        print(f'filename {f}\n\n')
        lingual_dict = np.load(f, allow_pickle=True).item()
        lingual_evaluation(lingual_dict, f)











