import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from openpyxl import load_workbook
import torch
import time
import openai

openai.api_key = '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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'



disease = pd.read_csv('../format_dataset.csv', header=None)
cols = [1, 2, 3, 4, 5]
combined_disease = disease.iloc[:, cols].astype(str).apply(' '.join, axis=1)
# print(combined_disease[1:])
context = ''
for i in range(3):
    context += combined_disease[1+i]
    context += '\n'

print(context)

prompt_trans = """Translate the following context into Chinese: \n""" + context
s_time = time.time()
response_dia = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": " "},
            {"role": "user", "content": prompt_trans},

        ]
)
dialogue_modi = response_dia['choices'][0]['message']['content']
e_time = time.time()

print(dialogue_modi)
print("Prompting time: ", e_time - s_time)










