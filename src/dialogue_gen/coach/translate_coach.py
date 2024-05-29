import pickle
import numpy as np
import pandas as pd
import glob
from translate import Translator
import openai
import time

# 163: dialogue_gen



FILE_PREFIX = 'coach_distance_match_20/coach_distance_20_*'
SAVE_PATH = 'translation_20/translated_'




file_paths = glob.glob(FILE_PREFIX)

coach_dic = {}
for file in file_paths:
    with open(file, 'rb') as f:
        d = pickle.load(f)
    coach_dic.update(d)

keylist = list(coach_dic.keys())
# print(keylist)
print(len(coach_dic))


# translator = Translator(from_lang='zh', to_lang='en')  # Translate to eng


def gpt_translate(context, retry=1, max_retry=5):
    prompt = """Translate the following text into English:\n""" + context

    try:
        coach_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85
        )
        response = coach_response['choices'][0]['message']['content']
        return response

    except openai.OpenAIError as e:
        if retry < max_retry:
            print(f'GPT error: {e}, retry={retry}')
            return gpt_translate(context, retry=retry + 1)
        else:
            print(f'GPT error retrying has reached limit.')
            exit(1)


translated_conversation = {}

idx = 2981 #2140 2580
end_idx = 3000
SAVE_FILE_NAME = SAVE_PATH + str(idx)
for key in keylist[idx:end_idx]:
    s_t = time.time()
    # print(key)
    # print(coach_dic[key])
    conversion = coach_dic[key]
    lines = conversion.splitlines()
    length = len(lines)
    for i in range(length):
        if lines[i].strip() == '':
            continue

        if lines[i].startswith('教练：'):
            if lines[i] == '教练：':
                if i+1 < length:
                    translation = gpt_translate(lines[i+1])
                    lines[i+1] += translation

                # print(lines[i+1])
            else:
                translation = gpt_translate(lines[i])
                lines[i] += translation
                # print(lines[i])
    new_conversation = '\n'.join(lines)

    # e_t = time.time()
    # print(key)
    # print('\n')
    # print('Running time: ', e_t - s_t)
    # print(new_conversation)
    translated_conversation[key] = new_conversation
    idx += 1
    if idx % 20 == 0:
        print(idx)
        print(key)
        print(translated_conversation[key])
#
        with open(SAVE_FILE_NAME, 'wb') as f:
            pickle.dump(translated_conversation, f)
















