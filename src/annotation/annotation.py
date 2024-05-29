
import random
import pickle
import numpy as np
import pandas as pd
import glob
from translate import Translator
import openai
import time
from inference import Agent
from transformers import AutoTokenizer
import csv

label = ['医生：', '病人：', '教练：']

with open('Embedding/disease_all', 'rb') as f:
    disease = pickle.load(f)
with open('Embedding/disease_name', 'rb') as f:
    disease_name = pickle.load(f)
disease_name = disease_name.reshape(-1, 1)

Con_agent = Agent(3)
### Extracting patient and doctor and dialogue_gen history


mediacl_label = ['医疗知识：']
conv_label = ["对话："]
machine_label = ["机器标注："]
human_label = ["人工标注："]


def utterance_extraction(conversation):
    address = [[], [], []]

    utterances = conversation.splitlines()
    for i, lab in enumerate(label):
        for idx, utterance in enumerate(utterances[:-1]):
            ### doctor detection
            if utterance.startswith(lab):
                ### single label line
                if utterance == lab:
                    sen = utterances[idx+1]
                    address[i].append(sen)
                ### label + utterance line
                else:
                    sen = utterance.split(lab)[1]
                    # print(sen)
                    address[i].append(sen)

    return address


def annotation_dataset(match_dic, coach_dic, annotation):
    coach_input = []
    coach_target = []
    annotation_case = ''
    case_idx = 0
    annote_idx = 0

    for key in list(coach_dic.keys()):
        case_idx += 1
        # print(key)

        address = utterance_extraction(coach_dic[key])
        current_disease = match_dic[key]
        current_disease, medical = Con_agent.select_context(current_disease, disease_name, disease)
        medical = medical[0]

        min_len = min(len(address[0]), len(address[2]))
        history = []
        case_label = f'\n\nCase: {case_idx}'
        case = [case_label, mediacl_label[0], medical, conv_label[0]]

        for i in range(min_len):
            current_doctor = label[0] + address[0][i]

            try:
                current_patient = label[1] + address[1][i]
            except IndexError:
                # current_patient = ''
                continue
            current_coach = label[2] + address[2][i]

            # coach_prompt = Con_agent.coach_prompt(medical, current_doctor, '\n'.join(history))
            # print(annotation[annote_idx])
            current_case = [current_doctor, current_coach, machine_label[0] + annotation[annote_idx][1] + '\n' + human_label[0], current_patient]

            case += current_case
            annote_idx += 1

            if annote_idx == 2170:
                annotation_case += '\n\n'.join(case)
                return annotation_case
            # print(annote_idx)

        # print('\n\n'.join(case))
        annotation_case += '\n\n'.join(case)
    return annotation_case



if __name__ == '__main__':

    annota_file = 'annotation_data/gpt_coach.csv'
    text_file = 'annotation_data/gpt_coach.txt'

    annotation = []
    with open(annota_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                annotation.append(row)
    annotation = annotation[1:]
    print(len(annotation))

    with open('parsed_2020t', 'rb') as f:
        dic = pickle.load(f)

    with open('match_key_disease_dic20', 'rb') as f:
        match_case = pickle.load(f)

    test_key = np.load('../test_index.npy')
    print('test_key:', test_key.shape)
    print(test_key)

    FILE_PREFIX = '../../Dialogue_gen/coach/coach_20/coach_20_*'

    file_paths = glob.glob(FILE_PREFIX)

    coach_dic = {}
    for file in file_paths:
        with open(file, 'rb') as f:
            d = pickle.load(f)
        coach_dic.update(d)

    keylist = list(coach_dic.keys())
    # print(keylist)
    print(len(coach_dic))
    # print(len(match_case))

    test_dic = {}
    for key in test_key:
        test_dic[key] = coach_dic[key]

    print(len(test_dic))

    annotation_txt = annotation_dataset(match_case, test_dic, annotation)

    with open(text_file, 'w') as f:
        f.write(annotation_txt)


