import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai

## 163 gcot
openai.api_key = ''

eval_key = np.load('../../annotation/eval_keys.npy')

### get lens of eval key
with open('../../annotation/non_lingual/data/coach_dict/no_lora_dict_test', 'rb') as f:
    dic = pickle.load(f)


coach_input = []
with open('../coach_data/gcot_coach.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            coach_input.append(row)

coach_input = coach_input[1:]

eval_lens = {key: len(value) for key, value in dic.items()}


coach_dict = {}
start_index = 0
end_idx = 0
for key in eval_lens.keys():
    lens = eval_lens[key]
    end_idx += lens
    # no_lora_anno_dict[key] = no_lora_annotation[start_index:end_idx]
    # no_lora_coach_dict[key] = no_lora_coach[start_index:end_idx]

    coach_dict[key] = coach_input[start_index:end_idx]

    start_index = end_idx

with open('gcot_coach_dict.csv', 'wb') as f:
    pickle.dump(coach_dict, f)

with open('gcot_coach_dict.csv', 'rb') as f:
    coach_input_dict = pickle.load(f)


def gpt_call(prompt, retry=0):
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
        print(e)
        print('retry:', retry)
        return gpt_call(prompt, retry=retry+1)



i = 0
coach_gen = {}
for key, value in coach_input_dict.items():
    coach_gen[key] = []
    for sen in value:
        gen = gpt_call(sen[0])
        coach_gen[key].append(gen)
        i += 1

        if i % 10 == 0:
            with open('gcot_gen.csv', 'wb') as f:
                pickle.dump(coach_gen, f)
            print(i)
            print(gen)

with open('gcot_gen.csv', 'wb') as f:
    pickle.dump(coach_gen, f)




print(eval_lens)









