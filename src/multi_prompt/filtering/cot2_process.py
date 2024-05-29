import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os
import re

from multi_prompt.pipeline.run import det_cor_gen, gpt_call4, lingual_detect_prompt


# new model with original prompt filtering
def filtering_ori(sen):
    sen_split = re.split('[{Output:}]', sen)
    if len(sen_split) != 0:
        return sen_split[-1]

    else:
        lines = [l for l in sen.splitlines() if l.strip()]
        return lines[-1]

def filtering_none(sen):
    lines = [l for l in sen.splitlines() if l.strip()]
    if len(lines) < 12 or ('P' not in sen):
        return 'None'
    else:
        return lines[-1]




if __name__ == "__main__":
    cot2 = np.load('cot2_generated.npy', allow_pickle=True).item()

    with open('../../human_label_data/lingual_pos', 'rb') as f:
        new_lingual_pos = pickle.load(f)

    new_lingual_pos = {int(key): value for key, value in new_lingual_pos.items()}


    cot_coach = {}
    cot_list = []
    none_num = 0
    for key, v_list in cot2.items():
        # key = int(key)
        cot_coach[key] = []
        for v in v_list:

            # sen = bcot[key][v]
            sen = v
            coach = filtering_ori(sen)
            if coach == 'None':
                none_num += 1
                print(key, v)
            cot_coach[key].append(coach)
            cot_list.append(coach)

## 163
    openai.api_key = ''

    FILENAME = 'cot2'

    # FILE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    COACH_DIC = cot_coach

    ### lingual
    LINGUAL_DICT = det_cor_gen(coach_dict=COACH_DIC, filename=FILENAME)








