import random
import pickle
import numpy as np
import pandas as pd
import glob
from translate import Translator
import openai
import time
from Testing.inference import Agent
from transformers import AutoTokenizer
import csv

###-------------------------------------------------------------------------------------------------------------------
# Get the number of utterances of each element in coach_dic corresponding to a conversation case
# Test key has 500 samples
# Eval key has 300 samples which are current testing set
# Return a dict{key: utterance number}


# Reorganize the coach sentence csv file and make it separated based on key(number of case)




label = ['医生：', '病人：', '教练：']
with open('../../../Embedding/disease_all', 'rb') as f:
    disease = pickle.load(f)
with open('../../../Embedding/disease_name', 'rb') as f:
    disease_name = pickle.load(f)
disease_name = disease_name.reshape(-1, 1)

# Con_agent = Agent(3)
### Extracting patient and doctor and dialogue_gen history

with open('../../../parsed_2020t', 'rb') as f:
    dic = pickle.load(f)

with open('../../../Dialogue_gen/coach/key_extraction/match_key_disease_dic20', 'rb') as f:
    match_case = pickle.load(f)

test_key = np.load('../../test_index.npy')
print('test_key:', test_key.shape)

FILE_PREFIX = '../../../Dialogue_gen/coach/coach_20/coach_20_*'

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



def key_utterances(match_dic, coach_dic):
    coach_input = []
    coach_target = []
    annotation_case = ''
    case_idx = 0
    annote_idx = 0

    key_utterance = {}

    for key in list(coach_dic.keys()):
        case_idx += 1
        # print(key)

        address = utterance_extraction(coach_dic[key])
        current_disease = match_dic[key]
        # current_disease, medical = Con_agent.select_context(current_disease, disease_name, disease)
        # medical = medical[0]

        min_len = min(len(address[0]), len(address[1]), len(address[2]))
        key_utterance[case_idx] = min_len

        for i in range(min_len):
            current_doctor = label[0] + address[0][i]

            try:
                current_patient = label[1] + address[1][i]
            except IndexError:
                # current_patient = ''
                continue
            current_coach = label[2] + address[2][i]

            annote_idx += 1
    return key_utterance


if __name__ == '__main__':

    utterances = key_utterances(match_case, test_dic)



    ### loading csv file of human labeled coach sentence.
    human_coach = []
    with open('../training_annotation/coach_human.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if row:
               human_coach.append(row)

    print(len(human_coach))




    ### loading csv.file of no lora coach sentences
    no_lora_coach = []
    with open('../coach_data/alpaca_no_lora.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if row:
                no_lora_coach.append(row)
    no_lora_coach = no_lora_coach[1:]
    print(len(no_lora_coach))


    human_coach_dict = {}
    no_lora_annotation = np.load('../annotation_data/no_lora.npy')
    no_lora_anno_dict = {}
    no_lora_coach_dict = {}
    start_index = 0
    end_idx = 0
    for key in utterances.keys():
        lens = utterances[key]
        end_idx += lens
        # no_lora_anno_dict[key] = no_lora_annotation[start_index:end_idx]
        # no_lora_coach_dict[key] = no_lora_coach[start_index:end_idx]

        human_coach_dict[key] = human_coach[start_index:end_idx]

        start_index = end_idx



    eval_key = np.load('../eval_keys.npy')

    # no_lora_anno_dict_test = {key: no_lora_anno_dict[key] for key in eval_key}

    # with open('data/coach_dict/no_lora_dict_test', 'wb') as f:
    #     pickle.dump(no_lora_anno_dict_test, f)
    #
    # with open('data/coach_dict/no_lora_coach_dict', 'wb') as f:
    #     pickle.dump(no_lora_coach_dict, f)

    with open('data/coach_dict/human_coach_dict', 'wb') as f:
        pickle.dump(human_coach_dict, f)













