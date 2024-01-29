import openai
import numpy as np
import pickle
import json
import pandas
import interface.Embedding.dialogue_embed as embed


#-----------------------------------------------------------------------------------------------------#
# 1. Medical knowledge extraction -> disease base loading -> find in the name list at first -> calculate embedding distance
#
# 2. Human doctor: Input string capture
#
# 3. Coach AI: prompt -> Coach sentences storing not interfering patient-doctor dialogue
#
# 4. Patient AI: Prompt -> Completing conversation
#
#-----------------------------------------------------------------------------------------------------#


# 163 con_agent


coach_head_prompt = """Your role is to act as a linguistic coach for a doctor, ensuring their medical advice aligns with the provided context. If discrepancies are identified in the doctor's dialogue compared to the provided medical context, guide them towards making more accurate statements.
You need to perform the following actions with the specific order:
1.You need detect the <medical words> in the <doctor's statement>. 
2.Then you need to make comparison between <medical words> with <medical context>.
3.If all of <medical words> in <doctor's statement> are relevant to <medical context>, then just respond something encouraging.
4.If the doctor makes errors regarding symptom descriptions, respond with:
"Doctor, there seems to be a discrepancy regarding symptom descriptions. The term '<incorrect symptom words>' isn't appropriate for '<current disease>'. Perhaps you should consider '<correct symptoms>'."
5.For errors related to disease or medication names, use this structure:
"Doctor, there's an inconsistency with the medical term. '<incorrect words>' isn't right. The appropriate term would be '<correct words>'."

If sentence contains multiple errors, you need to point all of them out based on above patterns.

Please remember that '<incorrect symptom words>' and '<incorrect words>' refer to the misused words you detected. '<current disease>' and '<correct symptoms>' are derived from the medical context provided. 
Please replace these placeholders with correct medical terms from <medical context> based on above instruction.
Your response also need to based on <dialogue history> to make your response more suitable in dialogue, but only perform above actions on <doctor's statement>
Please show me Chinese response directly.

Provided medical context: 
<medical context>: """

# coach_prompt = """\n<Misuse of medical terms>: Those medical terms in doctor sentence are not relevant to the provided <medical knowledge>
# Suppose you are a medical and language coach which means you need to speak in coach tone.Now, detect the <misuse of medical terms> in the following doctor sentence and show me response directly:\n"""



patient_head_prompt = """Based on the <patient profile> and <dialogue history> provided below, predict the next sentence the patient would likely say in a patient-doctor conversation. Provide the response in the format: 病人：[Your Response Here].

<patient profile>: \n"""


# print(coach_prompt)

### Agent would generate response from coach or patient based on provided medical context and patient profile
class Agent:
    def __init__(self, model):
        self.model = 'gpt-4' if model == 4 else 'gpt-3.5-turbo'
        # self.dic_all = dic_all
        # print(self.model)

    ### if name not in disease_name, select the most similar disease based on embedding
    def select_context(self, name, disease_name, disease_all):
        if name in disease_name:
            key = np.where(disease_name == name)[0]
            return name, disease_all[key]
        else:
            disease_embedding = np.load('Embedding/match_disease_embedding.npy')
            dia_embed = embed.get_embedding(name)
            max_index = embed.calculating_distance(dia_embed, disease_embedding)[-1]
            new_name = disease_name[max_index]
            print(max_index)
            print(new_name)
            return new_name, disease_all[max_index]

    def profile(self, dic_all, match_dic, disease):

        # disease = '声带息肉'
        disease_key = 0

        for key, value in match_dic.items():
            if value == disease:
                disease_key = key
                # print(disease_key)
                # print(match_dic[disease_key])
                break
        profile = dic_all[disease_key]['Description']
        # print(profile)
        return profile

    def response_generation(self, prompt, regen=1, max_regen=5, retry=1, max_retry=5):
        try:
            coach_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                # api_key=self.api_key
            )
            response = coach_response['choices'][0]['message']['content']


            if self.check_response(response):
                # print('Generate successfully')
                return response
            elif regen < max_regen:
                # print('regeneration\n')
                return self.response_generation(prompt, regen=regen + 1)
            else:
                print("Regeneration has reached limit")
                return ''

        except openai.OpenAIError as e:
            if retry < max_retry:
                print(f'GPT error: {e}, retry={retry}')
                return self.response_generation(prompt, retry=retry + 1)
            else:
                print(f'GPT error retrying has reached limit.')
                exit(1)

    def check_response(self, response):
        if '<' in response:
            return False
        else:
            return True

    def patient_prompt(self, profile, history):
        # profile = '\n'.join(profile)

        prompt = patient_head_prompt + profile + \
            """\n<dialogue history>:\n""" + history
        return prompt

    def patient(self, profile, history):
        # profile = '\n'.join(profile)

        prompt = patient_head_prompt + profile + \
            """\n<dialogue history>:\n""" + history



        response = self.response_generation(prompt)
        return response

    def coach_prompt(self, medical_knowledge, doctor_sen, history):
        prompt = coach_head_prompt + medical_knowledge + \
                 """\n<dialogue history>:""" + history + \
                 """\n<Doctor's statement>:""" + doctor_sen + \
                 """\nYour response (Generated response in Chinese):"""

        return prompt

    def coach(self, medical_knowledge, doctor_sen, history):
# Chain of thought
### detection function for error is fine but it also provided correction for correct sentences
### Currently best prompt, capable of handling multiple errors and discrimante correct and incorrect words.
### Note that put extra info and context at the last of prompt.
### Position of same content in prompt does affect the response
        prompt = coach_head_prompt + medical_knowledge + \
                """\n<dialogue history>:""" + history + \
                """\n<Doctor's statement>:""" + doctor_sen + \
                """\nYour response (Generated response in Chinese):"""

        response = self.response_generation(prompt)
        return response









