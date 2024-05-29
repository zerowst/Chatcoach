from nltk.tokenize import sent_tokenize
from openpyxl import load_workbook
import torch
import openai
import pickle
import numpy as np
import time
import pandas as pd


### loading all dialogues


def get_embedding(text, model="text-embedding-ada-002", retry = 1, max_retry = 5):
    text = text.replace("\n", " ")
    try:
        embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        # print('Embedding successfully.')
        return embedding
    except openai.OpenAIError as e:
        if retry < max_retry:
            print(f'Embedding error: {e}, retry: {retry}')
            return get_embedding(text, model="text-embedding-ada-002", retry=retry+1)
        else:
            print(f'Embedding error retrying has reached limit.')
            exit(1)


def calculating_distance(dialogue_embed, disease_embed):
    distance = []
    for disease in disease_embed:
        dot_product = np.dot(dialogue_embed, disease)
        norm1 = np.linalg.norm(dialogue_embed)
        norm2 = np.linalg.norm(disease)
        similarity = dot_product / (norm1 * norm2)
        distance.append(similarity)
        # np.argsort(distance)

    return np.argsort(distance)[-3:]

def combine_text(dic, key):
    # key = str(key)
    dialogue = dic[key]['Dialogue'] if 'Dialogue' in dic[key] else []
    diagnosis = dic[key]['Diagnosis and suggestions'] if 'Diagnosis and suggestions' in dic[key] else []
    description = dic[key]['Description'] if 'Description' in dic[key] else []
    text_list = dialogue + diagnosis + description
    text = ''
    for t in text_list:
        text += t

    return text

def main():


    with open('../parsed_2020t', 'rb') as f:
        dic = pickle.load(f)

    label = ['医生：', '病人：']

    ### Keys with dialogue_gen and 'Diagnosis and suggestions'
    dic_both = []
    for key in dic.keys():
        if 'Dialogue' in dic[key].keys() and 'Diagnosis and suggestions' in dic[key].keys():
            dic_both.append(key)

    ### load disease embedding
    disease_embedding = np.load('disease_embedding.npy')
    print(disease_embedding.shape)
    # print(disease)

    cols = [1, 2, 3, 4, 5]
    disease = pd.read_csv('disease.csv', header=None)
    disease = disease.iloc[:, cols].astype(str).apply(' '.join, axis=1)
    disease = disease[1:]

    key = 4
    start = time.time()

    conver_text = combine_text(dic, str(key))
    dia_embed = get_embedding(conver_text)

    end = time.time()
    print("Dialogue embedding time: ", end - start)

    max_index = calculating_distance(dia_embed, disease_embedding)
    context = disease_embedding[max_index]
    end2 = time.time()
    print("Finding closest context time: ", end2 - end)
    print(context)

    print("Dialogue: ")
    print(dic[str(key)]['Dialogue'])
    print("Context: ")
    print(disease[max_index])
    print(max_index)


### Selecting context-------------------------------------------------
if __name__ == '__main__':

    # main()
    exit()






