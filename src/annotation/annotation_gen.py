import openai
import os
import pickle
import csv
import numpy as np
import time

openai.api_key = ''


annotation_prompt = """
Examples:
教练：医生，您的回答是正确的。但是，关于药物名称方面，可以更准确一些。您使用的“咽炎片”是错误的
错误术语：咽炎片 修改为：None

教练：医生，关于疾病名称的使用似乎不准确。术语肺炎并不准确。正确的术语应该是咽喉炎。请在回答中使用该术语。此外，对于治疗措施，您可以提供更具体的建议。例如，您可以说：“针对咽喉炎，我们可以考虑使用抗生素进行治疗。”
错误术语：肺炎 修改为：咽喉炎

教练：医生，关于药膏名称使用不准确。正确的药膏名称是红霉素眼膏或者是氧氟沙星眼膏。此外，如果有脓液的话，不需要活检排脓，而是需要进行穿刺排脓。如果没有化脓，可以先涂鱼石脂软膏促进化脓
错误术语：None  修改为：红霉素眼膏或者是氧氟沙星眼膏

Following above examples to do the annotation for given sentence. Detect the incorrect medical terms and corrected terms which original sentence want to present.
Detection for incorrect terms and correction are independent.
If you find any, please list them using the following format: 
错误术语：[Incorrect Term] 修改为：[Corrected Term].
If there are no incorrect terms, indicate this by stating [Incorrect Term] is None. Likewise, if there are no suitable corrections, please indicate [Corrected Term] is None. 
If multiple terms are incorrect, list all the incorrect terms and their corrections following the same format.
"""

def get_response(sentence):
    prompt = annotation_prompt + '\n' + sentence
    coach_response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        # api_key=self.api_key
    )
    response = coach_response['choices'][0]['message']['content']
    return response

def get_annotation(annotation_file, coach_list):

    if os.path.exists(annotation_file):
        annotation = list(np.load(annotation_file))

        i = len(annotation)
        print(f'Continue from {i}')
        i = i
    else:
        print('Annotating starts...')
        i = 0
        annotation = []

    for text in coach_list[i:]:
        try:
            note = get_response(text)
        except openai.OpenAIError as e:
            print(e)
            print(i)
            time.sleep(40)
            get_annotation(annotation_file, coach_list)
        annotation.append(note)
        # print(text)
        # print(note)
        if i % 10 == 0:
            print(i)
            print(note)
            np.save(annotation_file, annotation)


        i += 1

    np.save(annotation_file, annotation)


if __name__ == '__main__':
    coach_file = 'coach_data/gpt_coach.csv'
    annotation_file = 'annotation_data/gpt_coach.npy'

    llama_coach = []
    with open(coach_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                llama_coach.append(row)

    llama_coach = llama_coach[1:]
    llama_coach = [sen[0] for sen in llama_coach]

    print(len(llama_coach))

    get_annotation(annotation_file, llama_coach)




