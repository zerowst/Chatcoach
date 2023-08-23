import openai
import numpy as np
import pickle
import json
import pandas

# -----------------------------------------------------------------------------------------------------#
# 1. Medical knowledge extraction -> disease base loading -> find in the name list at first -> calculate embedding distance
#
# 2. Human doctor: Input string capture
#
# 3. Coach AI: prompt -> Coach sentences storing not interfering patient-doctor dialogue
#
# 4. Patient AI: Prompt -> Completing conversation
#
#
#
#
#
#
#
#
#
# -----------------------------------------------------------------------------------------------------#


# 163 con_agent
openai.api_key = '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'

coach_head_prompt = """Based on the provided <medical context> :
<medical context>:\n"""

coach_prompt = """\n<Misuse of medical terms>: Those medical terms in doctor sentence are not relevant to the provided <medical knowledge>
Suppose you are a medical and language coach which means you need to speak in coach tone.Now, detect the <misuse of medical terms> in the following doctor sentence and show me response directly:\n"""

patient_head_prompt = """Based on history of patient-doctor dialogue, predict the next patient sentence. 
The form should be: 
病人：<>
Show me sentence directly.
History:\n"""


# print(coach_prompt)

class Agent:
    def __init__(self, model):
        self.model = 'gpt-4' if model == 4 else 'gpt-3.5-turbo'
        print(self.model)

    def select_context(self, name):
        if name in disease_name:
            key = np.where(disease_name == name)[0]
            return disease[key]
        else:
            return None

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
                print('regeneration\n')
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
            print(response)
            return False
        else:
            return True

    def patient(self, history):
        prompt = patient_head_prompt + history
        response = self.response_generation(prompt)
        return response

    def coach(self, medical_knowledge, doctor_sen):
        prompt = coach_head_prompt + medical_knowledge + '\n' + coach_prompt + doctor_sen

        ### Chain of thought
        ### detection function for error is fine but it also provided correction for correct sentences
        ### Currently best prompt, capable of handling multiple errors and discrimante correct and incorrect words.
        ### Note that put extra info and context at the last of prompt.
        ### Position of same content in prompt does affect the response
        prompt = """Your role is to act as a linguistic coach for a doctor, ensuring their medical advice aligns with the provided context. If discrepancies are identified in the doctor's dialogue compared to the provided medical context, guide them towards making more accurate statements.

You need detect the medical terms in the <doctor's statement>. 
Then you need to make comparison between them with <medical context>.
If all of medical words in <doctor's statement> are relevant to <medical context>, then just respond something encouraging.
If the doctor makes errors regarding symptom descriptions, respond with:
"Doctor, there seems to be a discrepancy regarding symptom descriptions. The term '<incorrect symptom words>' isn't appropriate for '<current disease>'. Perhaps you should consider '<correct symptoms>'."
For errors related to disease or medication names, use this structure:
"Doctor, there's an inconsistency with the medical term. '<incorrect words>' isn't right. The appropriate term would be '<correct words>'."
If sentence contains multiple errors, you need to point all of them out based on above patterns.

Please remember that '<incorrect symptom words>' and '<incorrect words>' refer to the misused words you detected. '<current disease>' and '<correct symptoms>' are derived from the medical context provided. 
Please replace these placeholders with correct medical terms from <medical context> based on above instruction.
Please note, responses should be in Chinese.

Provided medical context: 
<medical context>: """ + medical_knowledge + \
"""\nDoctor's statement:""" + doctor_sen + \
"""\nYour response (Generated response in Chinese):"""



        response = self.response_generation(prompt)
        return response


if __name__ == '__main__':
    label = ['医生：', '病人：', '教练：']

    with open('disease_all', 'rb') as f:
        disease = pickle.load(f)
    with open('disease_name', 'rb') as f:
        disease_name = pickle.load(f)

    disease_name = disease_name.reshape(-1, 1)
    Con_agent = Agent(3)

    key = 1
    history = []
    coach = []
    # medical = disease[key]
    current_disease = input("疾病：\n")
    medical = Con_agent.select_context(current_disease)
    print(medical)
    print("--------------------------Patient background----------------------------")

    print("Patient has some medical issues\n")
    while True:
        current_doctor = input("医生：")
        history.append(label[0])
        history.append(current_doctor)

        # print(f'Medical: {medical[:5]} \n')
        print("Current doctor: ", current_doctor)

        current_coach = Con_agent.coach(medical, current_doctor)
        print("Current coach: ", current_coach)
        coach.append(current_doctor)

        current_patient = Con_agent.patient('\n'.join(history))
        # history.append(label[1])
        history.append(current_patient)

        print('Current patient: ', current_patient)
        print('Dialogue history: ', '\n'.join(history))
        print('-------------------turn over--------------------------\n\n')







