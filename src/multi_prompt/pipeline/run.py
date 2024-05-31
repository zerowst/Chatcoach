import openai
import pickle
import numpy as np
import csv
import os
import re
from evaluate import load
from annotation.metrics import qa_f1_zh_score, qa_f1_score
from multi_prompt.pipeline.coach_run import *
from multi_prompt.coach_gen.generate import *

# 163
openai.api_key = ''


def main():

    FILENAME = 'hybrid0'

    # generated response file path
    GEN_PATH = '../generated_coach/' + FILENAME + '_generated.npy'

    # model input file path
    INPUT_FILE = '../coach_data/' + FILENAME + '_coach.csv'
    # INPUT_FILE = '../coach_data/cot_coach.csv'

    # generating response
    input_list = load_input_file(INPUT_FILE)
    input_dict = process_input_file(input_list)
    generate_coach(input_dict, GEN_PATH, name=FILENAME)

    # loading generated response
    COACH_DIC = np.load(GEN_PATH, allow_pickle=True).item()

    # lingual detection and correction
    LINGUAL_DICT = det_cor_gen(coach_dict=COACH_DIC, filename=FILENAME)
    print(LINGUAL_DICT)



if __name__ == '__main__':

    main()

    exit()


