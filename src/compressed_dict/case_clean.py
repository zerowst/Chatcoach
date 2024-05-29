import numpy as np
import pickle
import csv
import re


FILE = 'outputhuman_clean.txt'
with open(FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines if line.strip()]


def clean_label(sentence):
    pattern = r"错误术语(.*?)修改为(.*?)(?=\n|错误术语|$)"

    # Find all matches in the string
    matches = re.findall(pattern, sentence)

    # Separate the terms
    terms_detected = [match[0] for match in matches]  # 错误术语 terms
    terms_corrected = [match[1] for match in matches]  # 修改为 terms
    str_det = ''.join(terms_detected).strip().replace(' ', '')
    str_cor = ''.join(terms_corrected).strip().replace(' ', '')

    symbols = r'[：:。，、（）\n\s\[\]\.\\\'\'\'"或者,]'
    str_det = re.sub(symbols, '', str_det)
    str_cor = re.sub(symbols, '', str_cor)
    return str_det, str_cor


all_cases = {}
index = 0
number_i = 0
n_case = 0
list_case = []
n_doctor = 0
n_patient = 0
n_coach = 0
n_human = 0
case_i = 0
start_i = 0

wrong_case = []

### human label and case clean
for i, line in enumerate(lines):

    if line.startswith('Case: '):
        n_case = line.split('Case: ')[1]
        list_case.append(n_case)
        # n_case += 1
        all_cases[n_case] = {}

    if line.startswith('颈椎病   腹痛'):
        # print(line)
        wrong_case.append(n_case)

        # print(index)

    if line.startswith('医疗知识：'):
        all_cases[n_case]['medical'] = lines[i+1]
        all_cases[n_case]['con'] = {'d': [], 'c': [], 'p': []}
        all_cases[n_case]['detection'] = []
        all_cases[n_case]['correction'] = []

    if line.startswith('医生：'):
        all_cases[n_case]['con']['d'].append(line)

    if line.startswith('教练：'):
        all_cases[n_case]['con']['c'].append(line)

    if line.startswith('病人：'):
        all_cases[n_case]['con']['p'].append(line)

    if line.startswith('人工标注：'):
        human_start = line.split('人工标注：')[1]
        if human_start and not human_start.startswith('错误术语'):
            human_start = '错误术语' + human_start
        start_i = i

    if line.startswith('病人：') and start_i != 0:
        end_i = i
        human_label = human_start + ''.join(lines[start_i+1: end_i])
        det, cor = clean_label(human_label)
        if det == '' or cor == '':
            print(n_case)
            print(human_label)
        all_cases[n_case]['detection'].append(det)
        all_cases[n_case]['correction'].append(cor)
        start_i = 0


all_cases = {key: all_cases[key] for key in all_cases.keys() if key not in wrong_case}

### saving all cases info
with open('all_cases', 'wb') as f:
    pickle.dump(all_cases, f)


### lingual human label
lingual_pos = {}
non_lingual_pos = {}

l_i = 0
n_i = 0

for key in all_cases.keys():
    lingual_pos[key] = []
    non_lingual_pos[key] = []
    for i, (det, cor) in enumerate(zip(all_cases[key]['detection'], all_cases[key]['correction'])):
        if det == 'None' and cor == 'None':
            non_lingual_pos[key].append(i)
            n_i += 1
        else:
            lingual_pos[key].append(i)
            l_i += 1


with open('lingual_pos', 'wb') as f:
    pickle.dump(lingual_pos, f)
with open('non_lingual_pos', 'wb') as f:
    pickle.dump(non_lingual_pos, f)




















