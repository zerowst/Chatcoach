import numpy as np
import pickle
import os
import openai
import codecs
import time


# gmail: gen
# openai.api_key = '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'

# 163: dialogue
openai.api_key = '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'


# dia = np.load('parsed_2020.npy', allow_pickle=True)

### load data
with open('parsed_2020t', 'rb') as f:
    dic = pickle.load(f)

### extract dialogue
# print(dic['4']['Dialogue'][0])
dialogue = dic['4']['Dialogue']
dialogue_s = ''
for d in dialogue:
    dialogue_s += d + '\n'

dialogue_ = dic['5']['Dialogue']
dialogue_fi = ''
for d in dialogue_:
    dialogue_fi += d + '\n'



### prompt setting
# 1. extract doctor sentences




### medical context
prompt_context = """Given the following text:
{恐慌症，["焦虑和紧张","抑郁症","呼吸急促","抑郁或精神症状","胸部剧烈疼痛","头晕","失眠","异常的不自主运动","胸闷","心悸","心律不齐"," \ 
呼吸加快"]，恐慌症是一种以反复发作的严重恐慌发作为特征的焦虑症。它可能还包括至少持续一个月的显著行为改变，并对其它发作的担忧或关注。后者 \ 
被称为预期性发作（DSM-IVR）。恐慌症与广场恐惧症（对公共场所的恐惧）不同，尽管许多患有恐慌症的人也患有广场恐惧症。恐慌发作无法预测，因此个体 \ 
可能会感到压力、焦虑或担心，想知道下一次恐慌发作将何时发生。恐慌症可以区分为医学状况或化学失衡。DSM-IV-TR将恐慌症和焦虑描述得不同。焦虑在慢 \
性压力因素的前提下逐渐累积，表现为中等强度的反应，可以持续数天、数周或数月，而恐慌发作是由突然而来的、出乎意料的原因引发的急性事件：持续时间短 \
且症状更加剧烈。恐慌发作可能发生在儿童和成人身上。对于年轻人来说，恐慌可能特别令人痛苦，因为孩子往往对所发生的事情了解较少，当发作发生时，\
父母也很可能感到困扰。，["心理治疗","心理健康咨询","心电图","抑郁筛查（抑郁症筛查）","毒物学筛查","心理和精神评估和治疗"]，["劳拉西泮",\
"阿普唑仑（Xanax）","氯硝西泮","帕罗西汀（Paxil）","文拉法辛（Effexor）","米氮平","丁螺酮（Buspar）","氟伏沙明（Luvox）","去甲噻嗪",\
"泮洛西汀（Pristiq）","氯米帕明","阿卡姆普罗酸（Campral）"]}""" + '\n'


### Conversation----
prompt_gen = prompt_context + \
"""Based on above context, create some misuse of medical words for doctor sentences on following conversation: \n""" + \
dialogue_s + '\n\n' + \
"""Your task:
1. For sentences from 医生: you need to replace some medical words with others from above given context 
2. For sentences from 病人: you should keep those sentences as they are now.

show me modifed conversation directly """

prompt_test = """who are you"""

s_time = time.time()
response_dia = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": " "},
            {"role": "user", "content": prompt_gen},

        ]
)
dialogue_modi = response_dia['choices'][0]['message']['content']

e_time = time.time()

print(dialogue_modi)

print("Prompting time: ", e_time - s_time)



