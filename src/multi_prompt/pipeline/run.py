import openai
import pickle
import numpy as np
import csv
import os
import re
from evaluate import load
import jieba
from annotation.metrics import qa_f1_zh_score, qa_f1_score, normalize_zh_answer, normalize_answer




def lingual_detect_prompt(coach_sen):
    prompt = """
Prompt:
Please analyze the given "coach sentence" to identify any incorrect medical terms and their respective corrections. \
Filter out any information unrelated to the identification and correction of medical terms, including any continuation \
of conversations, irrelevant words, or nonsensical parts.

Instructions:

The "coach sentence" will highlight errors in a doctor's statement and provide the corrected medical terms.\
 Extract these details and represent them as follows:

Single term discrepancy: 错误术语：[Incorrect Term] 修改为：[Corrected Term].
Multiple term discrepancies: 错误术语：[Incorrect Term1] [Incorrect Term2] 修改为：[Corrected Term1] [Corrected Term2].

In the absence of a detectable [Incorrect Term], format it as:
错误术语：None 修改为：[Corrected Term].
If no [Corrected Term] can be identified, format it as:
错误术语：[Incorrect Term] 修改为：None.
If no [Corrected Term] and [Incorrect Term] can be identified, format it as:
错误术语：None 修改为：None.

coach sentence:
""" + str(coach_sen) + \
"""
\nShow me results directly without any other words"""

    return prompt


def non_lingual_prompt(coach_sen, disease):
    prompt_head = """Instructions:
You will be provided with a sentence spoken by a coach, referred to as {coach sentence}, and a specific disease, \
referred to as {disease}. Based on these, you need to make two judgments:

1. Determine if the {coach sentence} contains encouragement or positive feedback for doctor's statement.

2. Determine if the {coach sentence}  provides explicit medical guidance from a coach to a doctor, relevant to the specified {disease}.\
Note that continuous writing of a conversation should be counted as False.

Input:
{coach sentence}: 
""" + str(coach_sen) + \
"""
{disease}: """ + str(disease) + \
"""

Output:
Provide your judgments in the following format:
xxx, xxx
Note: For both judgments, answer with True if the condition is met, otherwise answer with False. \
Show the Output directly without other words"""

    return prompt_head



def extract_det_cor(sentence):
    pattern = r"错误术语：(.*?) 修改为：(.*?)(?=\n|错误术语|$)"

    # Find all matches in the string
    matches = re.findall(pattern, sentence)

    # Separate the terms
    terms_detected = [match[0] for match in matches]  # 错误术语 terms
    terms_corrected = [match[1] for match in matches]  # 修改为 terms
    str_det = ''.join(terms_detected).strip().replace(' ', '')
    str_cor = ''.join(terms_corrected).strip().replace(' ', '')

    symbols = r'[：。，、（）\n\s\[\]\.\\\'\'\'"或者,]'
    str_det = re.sub(symbols, '', str_det)
    str_cor = re.sub(symbols, '', str_cor)

    return str_det, str_cor


def gpt_call4(prompt, retry=0):
    try:
        coach_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9
        )
        response = coach_response['choices'][0]['message']['content']

        return response

    except openai.OpenAIError as e:
        print(e)
        print('retry:', retry)
        return gpt_call4(prompt, retry=retry+1)


def non_lingual_filtering(coach_dict, filename):
    with open('../disease_dic', 'rb') as f:
        disease_dic = pickle.load(f)

    with open('non_lingual_pos', 'rb') as f:
        non_lingual_pos = pickle.load(f)

    saving_path = 'non_lingual_dic/' + str(filename) + '.npy'
    if os.path.exists(saving_path):
        non_lingual_filter_result = np.load(saving_path, allow_pickle=True).item()
        n_key = len(non_lingual_filter_result) - 1
    else:
        n_key = 0
        non_lingual_filter_result = {}

    if len(non_lingual_filter_result) == len(non_lingual_pos):
        return non_lingual_filter_result

    local_non = 0
    for key in list(non_lingual_pos.keys())[n_key:]:
        non_lingual_filter_result[key] = []

        for i, coach_sen in enumerate(coach_dict[key]):
            ### determine if the coach sentence belongs to non_lingual data
            if i in non_lingual_pos[key]:
                prompt_ = non_lingual_prompt(coach_sen=coach_sen, disease=disease_dic[key])
                gen = gpt_call4(prompt_)
                non_lingual_filter_result[key].append(gen)
                local_non += 1

                if local_non % 10 == 0:
                    np.save(saving_path, non_lingual_filter_result)
                    # with open(saving_path, 'wb') as f:
                    #     pickle.dump(vannila_gen, f)
                    print('non lingual data')
                    print(local_non)
                    print(gen)

    np.save(saving_path, non_lingual_filter_result)
    return non_lingual_filter_result


def det_cor_process(det_cor_dict):
    processed_det_cor = {'det': [], 'cor': []}
    for key, value in det_cor_dict.items():
        for v in value:
            det, cor = extract_det_cor(v)
            processed_det_cor['det'].append(det)
            processed_det_cor['cor'].append(cor)

    return processed_det_cor

# def human_lingual_process()

def non_lingual_evaluation(filter_result, filename):
    med = np.load('../../data_statistic/data/medical_advice.npy', allow_pickle=True).item()
    TT, TF, FT, FF = 0, 0, 0, 0
    med_advice = {}
    med_correct = 0
    for key in filter_result.keys():
        med_advice[key] = []
        for i, v in enumerate(filter_result[key]):
            if v == 'True, True':
                TT += 1
                if i in med[key].keys():
                    med_correct += 1
                    med_advice[key].append(i)
            elif v == 'True, False':
                TF += 1
            elif v == 'False, True':
                FT += 1
                if i in med[key].keys():
                    med_correct += 1
                    med_advice[key].append(i)
            elif v == 'False, False':
                FF += 1
            else:
                print(f'wrong case: key {key} value {v}')

    result = {'type': [TT, TF, FT, FF], 'med_correct': med_correct, 'med_advice': med_advice}
    print(result)

    saving_path = 'non_lingual_result/' + str(filename)
    np.save(saving_path, result)

    return result

def det_cor_gen(coach_dict, filename):

    with open('../../human_label_data/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    local_path = os.path.dirname(__file__)
    saving_path = local_path + '/lingual_dic/' + str(filename) + '.npy'
    if os.path.exists(saving_path):
        det_cor_dict = np.load(saving_path, allow_pickle=True).item()
        n_key = len(det_cor_dict) - 1
    else:
        n_key = 0
        det_cor_dict = {}

    if len(det_cor_dict) == len(lingual_pos):
        return det_cor_dict

    local_lin = 0

    for key in list(lingual_pos.keys())[n_key:]:

        det_cor_dict[key] = []

        for i, coach_sen in enumerate(coach_dict[key]):
            ### determine if the coach sentence belongs to lingual data
            if i in lingual_pos[key]:
                prompt_ = lingual_detect_prompt(coach_sen)
                gen = gpt_call4(prompt_)
                det_cor_dict[key].append(gen)
                local_lin += 1

                if local_lin % 10 == 0:
                    np.save(saving_path, det_cor_dict)
                    # with open(saving_path, 'wb') as f:
                    #     pickle.dump(vannila_gen, f)
                    print('lingual data')
                    print(local_lin)
                    print(gen)

    np.save(saving_path, det_cor_dict)
    return det_cor_dict


def MT_metrics(predictions, references):
    prediction_tokens = [list(jieba.cut(prediction, cut_all=False)) for prediction in predictions]
    ground_truth_tokens = [list(jieba.cut(ref, cut_all=False)) for ref in references]

    prediction_tokens = [[normalize_zh_answer(to) for to in token] for token in prediction_tokens]
    ground_truth_tokens = [[normalize_zh_answer(to) for to in token] for token in ground_truth_tokens]
    predictions = [' '.join(pre) for pre in prediction_tokens]
    references = [' '.join(true) for true in ground_truth_tokens]

    bleu = load("bleu")
    bleu_results = bleu.compute(predictions=predictions, references=references, max_order=2)

    rouge = load('rouge')
    rouge_result = rouge.compute(predictions=predictions, references=references)

    bert = load('bertscore')
    bert_result = bert.compute(predictions=predictions, references=references, lang='zh')
    bert_ave = sum(bert_result['precision']) / len(bert_result['precision'])
    return {'bleu': bleu_results, 'rouge': rouge_result, 'bert': bert_ave}
    # return {'bleu': bleu_results, 'rouge': rouge_result}


def Exact_Match(pre, truth):
    prediction_tokens = [list(jieba.cut(prediction, cut_all=False)) for prediction in pre]
    ground_truth_tokens = [list(jieba.cut(ground_truth, cut_all=False)) for ground_truth in truth]

    prediction_tokens = [[normalize_zh_answer(to) for to in token] for token in prediction_tokens]
    ground_truth_tokens = [[normalize_zh_answer(to) for to in token] for token in ground_truth_tokens]
    pred = [' '.join(pre) for pre in prediction_tokens]
    tru = [' '.join(true) for true in ground_truth_tokens]
    # prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    # ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]

    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=pred, references=tru)
    return results


def Accuracy(pre, truth):
    correct = 0
    for pred, ref in zip(pre, truth):
        if pred == ref:
            correct += 1
    return correct/len(pre)


def f1_score(pre, ref):
    score = 0
    lens = len(pre)
    if len(pre) == len(ref):
        for pre, ref in zip(pre, ref):
            score += qa_f1_zh_score(pre, ref)

    return score/lens


def lingual_evaluation(det_cor_dict, filename):
    ### loading human label
    with open('../../human_label_data/human_detection_all_old', 'rb') as f:
        human_detection = pickle.load(f)
    with open('../../human_label_data/human_correction_all_old', 'rb') as f:
        human_correction = pickle.load(f)
    with open('lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    lingual_human_detection = []
    lingual_human_correction = []
    for k, v in lingual_pos.items():
        for i in v:
            lingual_human_detection.append(human_detection[k][i])
            lingual_human_correction.append(human_correction[k][i])


    processed = det_cor_process(det_cor_dict)

    ### ACC
    det_acc = Accuracy(processed['det'], lingual_human_detection)
    cor_acc = Accuracy(processed['cor'], lingual_human_correction)

    ### EM
    # exact_match_metric = load("exact_match")
    # results = exact_match_metric.compute(predictions=processed['det'], references=lingual_human_detection)
    det_EM = Exact_Match(processed['det'], lingual_human_detection)
    cor_EM = Exact_Match(processed['cor'], lingual_human_correction)

    ### F1
    det_f1 = f1_score(processed['det'], lingual_human_detection)

    cor_f1 = f1_score(processed['cor'], lingual_human_correction)

    eval_dict = {'detection EM': det_EM, 'detection F1': det_f1, 'detection ACC': det_acc, 'correction EM': cor_EM,
                 'correction F1': cor_f1, 'correction ACC': cor_acc}

    ### MT metrics
    det_mt_results = MT_metrics(processed['det'], lingual_human_detection)
    cor_mt_results = MT_metrics(processed['cor'], lingual_human_detection)

    saving_path = 'lingual_result/' + str(filename)
    with open(saving_path, 'wb') as f:
        pickle.dump(eval_dict, f)







