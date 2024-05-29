import numpy as np
import pickle
import os
import openai
import codecs
import time
import pandas as pd
import Embedding.dialogue_embed as embed


# 163: dialogue_gen
openai.api_key = ''

# gpt4: misuse


#---------------------------------------------------------------------------------------------------------

def get_dialogue(key, dictionary_doc):
    dialogue_ori = dictionary_doc[str(key)]
    dialogue_try = ''
    for i in range(len(dialogue_ori)):
        dialogue_try += dialogue_ori[i]

    return dialogue_try


def get_medical_context(key, dictionary, context, disease_embedding):

    conver_text = embed.combine_text(dictionary, str(key))
    dia_embed = embed.get_embedding(conver_text)
    max_index = embed.calculating_distance(dia_embed, disease_embedding)
    medical_context = ''
    for idx in max_index:
        medical_context += context[idx + 1]

    return medical_context


def generation(medical_context, dialogue_try):

    prompt_context = """Given the following <medical context>:\n""" + medical_context + '\n'
    prompt_gen = prompt_context + \
    """Based on above context, create some cases misuse of medical words for doctor sentences in the following <dialogues>: \n""" + \
    dialogue_try + '\n' + \
    """ <Requirement for generation> YOU NEED TO FOLLOW:
    1.For sentences from 医生: you need to replace some medical words with medical words from above given <medical context>.
    2.Replacement of words should be extracted from <medical context>
    3.Only medical words need to be replaced. Other words should keep still.
    4.Your output should KEEP SAME PATTERN with given <dialogues>. DO NOT MISS any one of dialogue_gen of origin input. \n
    Here is your format of output:
    医生：
    <sentences>
    医生：
    <sentences>
    ...
    Until the end of given input of <dialogues>
    
    """
    try:
        response_dia = openai.ChatCompletion.create(


            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt_gen},
                ],
            temperature=0.85
        )
        dialogue_modi = response_dia['choices'][0]['message']['content']
        return dialogue_modi

    except openai.OpenAIError as e:
        print(f'Error occured: {e} ')
        return None

def regeneration(medical_context, dialogue_try, max_retries=5, delay_time=5):
    retries = 0
    while retries < max_retries:
        generated_response = generation(medical_context, dialogue_try)
        if generated_response is not None:
            print('Generation successfully.')
            return generated_response
        retries += 1
        print('Retrying: ', retries)
        time.sleep(delay_time)

    return None


if __name__ == '__main__':
    with open('../../parsed_2020t', 'rb') as f:
        dic = pickle.load(f)

    ### Keys with dialogue_gen and 'Diagnosis and suggestions'
    dic_both = []
    for key in dic.keys():
        if 'Dialogue' in dic[key].keys() and 'Diagnosis and suggestions' in dic[key].keys():
            dic_both.append(key)

    ### doctor sentences
    with open('../../doc_sen_doc', 'rb') as f:
        doc_dic = pickle.load(f)

    ### load disease embedding
    disease_embedding = np.load('../../Embedding/disease_embedding.npy')
    cols = [1, 2, 3, 4, 5]
    disease = pd.read_csv('../../Embedding/disease.csv', header=None)
    disease = disease.iloc[:, cols].astype(str).apply(' '.join, axis=1)
    disease_context = disease[1:]

    save_dic_misuse = {}

    i = 6601
    saved_filename = 'save_dic_misuse_' + str(i)

    doctor_keys = list(doc_dic.keys())[i:10000]

    for doctor_key in doctor_keys:
        if len(doc_dic[doctor_key]) == 0:
            print('Empty conversation')
            continue

        dialogue = get_dialogue(doctor_key, doc_dic)
        medical_context = get_medical_context(doctor_key, dic, disease_context, disease_embedding)
        response = regeneration(medical_context=medical_context, dialogue_try=dialogue)
        if response is None:
            print('Error: retry time has reached limit')
            exit(1)

        save_dic_misuse[doctor_key] = response

        if i % 10 == 0:
            with open(saved_filename, 'wb') as f:
                pickle.dump(save_dic_misuse, f)

            print("Saved number: ", i)
            print('\n')
            print(doctor_key)

            print("\nOrigin: \n", dialogue)
            print("\nModified: \n", response)
            print('\n')

        i += 1