import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os


# loading input file
def load_input_file(FILE):
    input_list = []
    with open(FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                input_list.append(row)
    input_list = input_list[1:]
    return input_list


def gpt_call(prompt, retry=0):
    try:
        coach_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        response = coach_response['choices'][0]['message']['content']

    except openai.OpenAIError as e:
        print(e)
        print('retry:', retry)
        response = gpt_call(prompt, retry=retry+1)

    return response


def gpt_icot(prompt, retry=0, regen=0):
    try:
        coach_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        response = coach_response['choices'][0]['message']['content']

    except openai.OpenAIError as e:
        print(e)
        print('retry:', retry)
        response = gpt_icot(prompt, retry=retry + 1, regen=regen)

    lines = [l for l in response.splitlines() if l.strip()]

    if len(lines) < 12 or ('P' not in response):
        if regen < 5:
            response = gpt_icot(prompt, retry=retry, regen=regen + 1)

    return response


def process_input_file(input_file):
    ### get lens of eval key
    with open('../../annotation/non_lingual/data/coach_dict/no_lora_coach_dict', 'rb') as f:
        dic = pickle.load(f)

    eval_key = np.load('../../human_label_data/eval_keys.npy')
    all_lens = {key: len(value) for key, value in dic.items()}

    input_dict = {}
    start_index = 0
    end_idx = 0
    for key in all_lens.keys():
        lens = all_lens[key]
        end_idx += lens

        input_dict[key] = input_file[start_index:end_idx]

        start_index = end_idx

    input_eval_dict = {k: input_dict[k] for k in eval_key}

    return input_eval_dict


def generate_coach(input_dict, saving_path, name):
    if 'bcot_ori' in name:
        gpt = gpt_icot
    else:
        gpt = gpt_call
    if os.path.exists(saving_path):
        gen_dict = np.load(saving_path, allow_pickle=True).item()
        n_key = len(gen_dict) - 1
    else:
        n_key = 0
        gen_dict = {}

    local_i = 0
    for key in list(input_dict.keys())[n_key:]:
        gen_dict[key] = []
        for sen in input_dict[key]:
            gen = gpt(sen[0])
            gen_dict[key].append(gen)
            local_i += 1

            if local_i % 10 == 0:
                np.save(saving_path, gen_dict)
                # with open(saving_path, 'wb') as f:
                #     pickle.dump(vannila_gen, f)
                print(local_i)
                print(gen)
    np.save(saving_path, gen_dict)



if __name__ == '__main__':


    FILENAME = 'cot2'
    SAVE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    INPUT_FILE = '../coach_data/' + FILENAME + '_coach.csv'
    # INPUT_FILE = '../coach_data/cot_coach.csv'

    input_list = load_input_file(INPUT_FILE)

    input_dict = process_input_file(input_list)

    generate_coach(input_dict, SAVE_PATH, name=FILENAME)











