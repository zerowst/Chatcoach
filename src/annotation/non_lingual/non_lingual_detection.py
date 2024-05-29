import openai
import os
import pickle
import csv
import numpy as np
import time
import re

### gmail key
openai.api_key = ''


###------------------------------------------------------------------
# new detection and correction for alpaca_no_lora
# none_machine_len: 473, none_human_len613, human_len1315





### prompt

def non_lingual_prompt(disease, coach):
    prompt_head = """Assess the given <coach sentence> to determine if it satisfies any of the following conditions:

    It provides explicit medical guidance from a coach to a doctor, relevant to the <specified disease>. It must not be continue writing.
    It validates a doctor’s statement as accurate or provides encouragement.

    Respond with 'True' if the sentence fulfills at least one condition; otherwise, 'False'. 
    
    Show me results directly without any other words.""" + '\n\n' + """specified disease: """ \
                  + str(disease) + """\n\ncoach sentence:\n\n""" + str(coach)

    return prompt_head

def gpt_call(prompt, retry=0):
    try:
        coach_response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            # api_key=self.api_key
        )
    except openai.OpenAIError as e:
        print(e)
        print('retry:', retry)
        time.sleep(10)
        return gpt_call(prompt, retry=retry+1)

    response = coach_response['choices'][0]['message']['content']
    return response



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


if __name__ == '__main__':
    ### disease list

    with open('../../../Embedding/disease_all', 'rb') as f:
        disease = pickle.load(f)
    with open('../../../Embedding/disease_name', 'rb') as f:
        disease_name = pickle.load(f)
    disease_name = disease_name.reshape(-1, 1)

    with open('../../../Dialogue_gen/coach/key_extraction/match_key_disease_dic20', 'rb') as f:
        match_case = pickle.load(f)

    with open('../human_detection_all', 'rb') as f:
        human_detection = pickle.load(f)

    with open('../human_correction_all', 'rb') as f:
        human_correction = pickle.load(f)

    test_key = np.load('../../test_index.npy')
    print('test_key:', test_key.shape)

    eval_key = np.load('../eval_keys.npy')
    print(eval_key.shape)

    ### loading coach sentence dict separated by key and case number
    with open('data/coach_dict/no_lora_dict_test', 'rb') as f:
        no_lora = pickle.load(f)

    with open('data/coach_dict/no_lora_coach_dict', 'rb') as f:
        no_lora_coach = pickle.load(f)

    processed_no_lora = {}
    for key, value in no_lora.items():
        processed_no_lora[key] = [[], []]
        for v in value:
            det, cor = extract_det_cor(v)
            processed_no_lora[key][0].append(det)
            processed_no_lora[key][1].append(cor)


    print('...')
    none_list = []
    none_det = []
    none_cor = []

    non_lingual_file = 'data/non_lingual_feedback.npy'

    print('Annotating starts...')

    non_lingual_feedback = []
    non_lingual = []
    idx = 0

    for key in processed_no_lora.keys():

        for i, (det, cor) in enumerate(zip(human_detection[key], human_correction[key])):
            if det == 'None' and cor == 'None':
                none_list.append(i)
                if processed_no_lora[key][0][i] == det:
                    none_det.append(i)
                if processed_no_lora[key][1][i] == cor:
                    none_cor.append(i)

                if processed_no_lora[key][0][i] == det and processed_no_lora[key][1][i] == cor:
                    non_lingual.append([key, i])
                    ### both human and machine label are none
                    # disease = disease_name[key]
                    disease = match_case[test_key[key-1]]
                    conv = no_lora_coach[key][i]
                    prompt = non_lingual_prompt(disease, conv)
                    feedback = gpt_call(prompt)
                    non_lingual_feedback.append(feedback)
                    if idx % 5 == 0:
                        np.save('data/non_lingual_feedback', non_lingual_feedback)
                        np.save('data/non_lingual', non_lingual)
                        print(idx)
                        print(disease)
                        print(conv)
                        print(feedback)
                    idx += 1

    np.save('data/non_lingual_feedback', non_lingual_feedback)
    np.save('data/non_lingual', non_lingual)

    # print(h_det_none/len(human_detection))
    none_all = set(none_det + none_cor)
    print(f'none_det_len:{len(none_det)}, none_cor_len:{len(none_cor)}, none_all_len{len(none_all)}')

    print(f'none_machine_len: {len(non_lingual)}, none_human_len{len(none_list)}, human_len{len(human_detection)}')







