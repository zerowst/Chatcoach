import openai
import numpy as np
import pickle
import json
import pandas
import interface.Embedding.dialogue_embed as embed
from interface.inference import Agent

openai.api_key = "YOUR OPENAI KEY"


def main():

    label = ['医生：', '病人：', '教练：']

    with open('match_key_disease_dic20', 'rb') as f:
        match_dic = pickle.load(f)
    with open('parsed_2020t', 'rb') as f:
        dic = pickle.load(f)
    with open('Embedding/disease_all', 'rb') as f:
        disease = pickle.load(f)
    with open('Embedding/disease_name', 'rb') as f:
        disease_name = pickle.load(f)
    disease_name = disease_name.reshape(-1, 1)


    # print(match_dic.values())
    match_list = list(match_dic.values())
    # print(match_list)
    # disease_name = match_list
    # print(np.unique(match_list))
    Con_agent = Agent(4)

    key = 1
    history = []
    coach = []
    # medical = disease[key]
    # print('start...')

    current_disease = input("请输入需要练习的疾病：\n")
    current_disease = current_disease.replace(' ', '')
    current_disease, medical = Con_agent.select_context(current_disease, disease_name, disease)
    medical = medical[0]
    print(medical)
    profile = Con_agent.profile(dic, match_dic, current_disease)
    # print(medical)
    input('\n下面是病人信息：')
    print('----------------------------Patient Profile-------------------------------')
    print('\n'.join(profile))
    print('--------------------------------------------------------------------------')

    # input('')
    print('---------------------------------对话开始-----------------------------------')

    while True:

        current_doctor = input("医生：")
        # print('\n')

        # print(f'Medical: {medical[:5]} \n')
        # print(Con_agent.coach_prompt(medical, current_doctor, '\n'.join(history)))

        current_coach = Con_agent.coach(medical, current_doctor, '')

        history.append(label[0] + current_doctor)
        print("\n医学教练：", current_coach)
        coach.append(current_doctor)
        input('')

        current_patient = Con_agent.patient('\n'.join(profile), '\n'.join(history))
        # history.append(label[1])
        history.append(current_patient)

        print(current_patient)
        input('')


if __name__ == '__main__':
    main()



