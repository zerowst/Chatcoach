import pickle
import csv
import re
import numpy as np




with open('../annotation/human_detection_all', 'rb') as f:
    human_detection = pickle.load(f)


def saving_file(input_, filename):
    with open('coach_data/' + str(filename) + '_coach' + '.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input'])
        for c in input_:
            writer.writerow([c])


def get_disease(text):
    pattern = r"<medical context>:\s*(\S+)"

    match = re.search(pattern, text)
    disease = match.group(1)
    return disease

### head prompt + doctor + medical + Output: Your response (Generated response in Chinese)
def replace_prompt(prompt, text):
    start = text.find("""Provided medical context:""") + len("""Provided medical context:""")
    end = text.find('<dialogue_gen history>:')
    start_ = text.find("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")

    new_text = prompt + text[start_: end_] + text[start:end] + """\nOutput: Your response (Generated response in Chinese):"""
    return new_text


### vannila prompt

vannila_prompt = """As a linguistic coach for a physician, evaluate the doctor's statement against the given medical context. \
If there are discrepancies, guide the doctor. If not, provide positive feedback.
"""

### zeroshot
zeroshot_prompt = """As a linguistic coach for a physician, evaluate the doctor's statement against the given medical context. \
If there are discrepancies, guide the doctor. If not, provide positive feedback. Please answer in Chinese and think step by step.
"""


### generalized cot
vcot_prompt = """Act as a linguistic coach for a physician. Assess the doctor's statement: {doctor’s statement}  against a provided medical context: {medical context}  and guide the physician if discrepancies arise. 
Your thought: the medical words in the statement are: xxxx.  By comparing xxx with the medical context,  if I find that xxx is not align with the {medical context}, since this is related to symptoms and it should be treated as incorrect symptoms, then I should check the corresponding accurate symptoms is ***. If I find that xxx is align with the {medical context}, then I should encourage the doctor and provide medical advice about xxx inside of the {doctor's statement}.
So, your final response:  
Doctor, 'xxx' isn't aligned with '<correct disease>'. Consider '<correct symptoms> 
Or something encouraging and medicl advice.
Note: <correct disease> and <correct symptoms> are extracted from {medical context}

Examples:
Input:
{doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：可能是气管炎。 建议行气管镜检查，了解气管部情况。

{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐  

Output: 教练：医生，根据病情资料和医学背景，您使用的术语气管炎不是很准确，正确的医学词汇应该是是咽喉炎。建议行电子咽喉镜检查，以了解咽喉部情况。

Input:
{doctor's statement}:医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。

{medical context}:高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波）     氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利  

Output:教练：医生，您提到的病情似乎有一些误解。在高血压的背景下，使用'尿路高血压'这个词似乎不太合适。您可以考虑使用以下术语：'肾性高血压'。另外，在提到药物时，使用'神经元'似乎不是正确的术语。您可能需要考虑使用正确的药物名称。

Input:
{doctor's statement}:医生：左氧氟沙星眼水一天六次。

{medical context}:麦粒肿   眼睛疼痛    眼睛肿胀    眼睛发红    眼睑肿胀    眼睛症状    眼睑肿块    眼睛发痒    眼睑病变或皮疹    异常出现皮肤   眼睛灼伤或刺痛   皮肤肿胀   流泪   外部麦粒肿或麦粒肿 /\xcb\x88sta\xc9\xaa/，也称为麦粒肿 /h\xc9\x94r\xcb\x88di\xcb\x90\xc9\x99l\xc9\x99m/，是皮脂腺的感染Zeis 在睫毛根部，或 Moll 的大汗腺感染。外部麦粒肿在眼睑外侧形成，可以看到是红色的小肿块。内部麦粒肿是眼睑内侧的睑板腺皮脂腺感染。它们还会在眼睑下方形成一个红色肿块，外部仅可见泛红和肿胀。麦粒肿与麦粒肿相似，但往往体积更小，疼痛更严重，而且通常不会产生持久的损伤。它们含有水和脓液，如果麦粒肿被用力弄破，细菌就会扩散。麦粒肿的特点是急性发作且持续时间通常较短（7\xe2\x80\x9310 天，未经治疗），而霰粒肿是慢性的，通常不经干预无法解决。麦粒肿通常由金黄色葡萄球菌引起。   物理治疗练习；操纵；和其他程序   切开和引流（I d）   眼科检查和评估（眼科检查）   非手术去除异物   培养伤口     红霉素 ， 红霉素眼药 ， 磺胺醋酸钠眼药 ， 庆大霉素眼药 ， 妥布霉素眼药 ， 妥布霉素（Tobi） ， 丁卡因（一触式） ， 地塞米松 - 妥布霉素眼药 ， 庆大霉素（庆大霉素）   四氢唑啉眼科   荧光素眼科  

Output:医生，您对于麦粒肿患者使用左氧氟沙星眼水一天六次的建议是恰当的。这显示了您对疾病的专业理解和治疗的精准把握。对于麦粒肿，确保患者严格遵守用药频率和疗程是非常重要的，以促进快速恢复。同时，提醒患者注意个人卫生，避免用手触摸眼部，这样可以减少感染的风险，并防止病情恶化。

Now assess the doctor's statement: {doctor’s statement}  against a provided medical context: {medical context}  and guide the physician if discrepancies arise. 
"""
# vcot_coach_input = [replace_prompt(vcot_prompt, t) for t in coach_input]
#
#
# with open('coach_data/hybrid_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in vcot_coach_input:
#         writer.writerow([c])



### vanilla chain of thought with four examples containing thinking steps
file = 'cot2'
prompt = """Act as a linguistic coach for a physician. Assess the doctor's statement: {doctor’s statement}  against a provided\
 medical context: {medical context}. You should provide your response based on the following examples of input, \
 thinking steps and output.

Examples:

{Input}:
{doctor’s statement}: 医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。
{medical context}: 高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波） 	氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利  

{thinking steps}:
作为一名针对医生的医疗指导教练，我会根据提供的医疗背景和医生的陈述来进行指导。以下是我的分步回应和指导：

核对医疗背景与医生陈述：

医疗背景：高血压（HTN）或称为动脉高血压，是一种动脉血压升高的慢性疾病。正常静息血压在100-140mmHg收缩压和60-90mmHg舒张压范围内。持续处于或高于140/90mmHg的血压被认为是高血压。治疗方法包括药物治疗（如氢氯噻嗪、氨氯地平、奥美沙坦或贝那普利）和生活方式的改变。重要的检测项目包括血液学测试、全血细胞计数、脂质面板、葡萄糖测量、心电图和超声检查等。

医生的陈述：医生提到“尿路高血压控制怎样。应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。”

分析：

医生提到的“尿路高血压”可能是指肾性高血压，这是高血压的一种特殊类型，与肾脏健康状况有关。
他提议的“吃这些神经元”表述不够明确，可能是指药物治疗，但需要具体化。
提到“血压控制不好最好医院面诊调药”是合理的，表示对病情的动态管理和调整治疗方案的重要性。
回应：

针对医生的陈述，建议如下：

明确“肾性高血压”的诊断需要通过具体的检测，如肾功能测试和肾脏超声波检查。
在提及药物治疗时，应详细说明具体的药物名称和剂量，以便于更准确的治疗和沟通。
您提到的“面诊调药”是非常重要的，持续监测和根据病情调整治疗方案是高血压管理的关键。

{Output}: 
医生，您提到的病情似乎有一些误解。在高血压的背景下，使用'尿路高血压'这个词似乎不太合适。您可以考虑使用以下术语：'肾性高血压'。另外，在提到药物时，使用'神经元'似乎不是正确的术语。您可能需要考虑使用正确的药物名称。


{Input}:
{doctor's statement}: 医生：别担心，积极治疗就好了。
{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐

{thinking steps}:
医疗情境分析及医生陈述：

医疗情境: 患者出现咽喉炎症状，包括喉咙痛、发烧、咳嗽、头痛、呕吐、耳痛、鼻塞、皮疹、全身酸痛、吞咽困难、发冷、食欲不振。这些症状表明患者可能患有链球菌性咽炎或链球菌性扁桃体炎，通常与A组链球菌感染相关。治疗方法可能包括流感病毒抗体检测、扁桃体切除术、阿莫西林、青霉素、头孢丙烯等药物治疗。

医生陈述: “别担心，积极治疗就好了。”

分析:
在这个情境中，医生的陈述基本符合医疗情境。医生鼓励患者不必过度担心，并强调了积极治疗的重要性。考虑到患者的症状和可能的链球菌感染，积极的治疗方法确实是关键。

无矛盾性：提供支持和相关医疗建议

鉴于医生的陈述与医疗情境相符，我建议您继续鼓励患者保持积极态度。同时，可以进一步解释治疗方法的重要性和效果，比如使用抗生素（如阿莫西林或青霉素）来对抗链球菌感染。此外，建议监测患者的症状变化，并在必要时调整治疗方案。若症状持续或加重，可能需要考虑进一步的检查或治疗，如扁桃体切除术。

{Output}:
医生，您的指导与当前医疗情境相符，您鼓励患者不必过度担心，同时强调了积极治疗的重要性。继续关注患者的反应和恢复情况，根据需要调整治疗方案，是保证患者健康的关键。您的专业态度和关怀对患者的恢复至关重要。

{Input}:
{doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：可能是气管炎。 建议行气管镜检查，了解气管部情况。

{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐  

{thinking steps}:
在这种情况下，您作为医疗教练需要帮助医生更准确地理解和处理病情。下面是对医生陈述和医疗背景的分析，以及相应的回应：

医疗背景和医生陈述：

医疗背景：患者表现出咽喉炎的症状，如喉咙痛、发烧、咳嗽、头痛、呕吐、耳痛、鼻塞、皮疹、全身酸痛、吞咽困难、发冷、食欲不振等。病因可能是链球菌性咽炎或链球菌性扁桃体炎，治疗包括抗生素（如青霉素）和对症治疗。
医生陈述：医生基于提供的病情资料，认为可能是气管炎，并建议进行气管镜检查。
分析：

存在的差异：医生的初步诊断是气管炎，这与病人展现的主要症状及潜在的链球菌性咽炎诊断不一致。气管炎通常表现为持续性咳嗽、呼吸困难等症状，而患者的主要症状是喉咙痛、发烧等，更符合咽喉炎或链球菌感染的特征。
回应：

指导医生重新考虑诊断：建议医生考虑链球菌性咽炎或链球菌性扁桃体炎的可能性，特别是考虑到患者的主要症状。建议进行相关的检测，如喉部拭子培养和抗链球菌抗体测试，以进一步确认诊断。
建议治疗方案的调整：鉴于当前的症状和可能的病因，建议采用适当的抗生素治疗（如青霉素或头孢类药物），并对症治疗以缓解患者的不适。

{Output}: 
医生，根据病情资料和医学背景，您使用的术语气管炎不是很准确，正确的医学词汇应该是是咽喉炎。建议行电子咽喉镜检查，以了解咽喉部情况。


{Input}:
{doctor's statement}:医生：左氧氟沙星眼水一天六次。
{medical context}:麦粒肿   眼睛疼痛    眼睛肿胀    眼睛发红    眼睑肿胀    眼睛症状    眼睑肿块    眼睛发痒    眼睑病变或皮疹    异常出现皮肤   眼睛灼伤或刺痛   皮肤肿胀   流泪   外部麦粒肿或麦粒肿，也称为麦粒肿 ，是皮脂腺的感染Zeis 在睫毛根部，或 Moll 的大汗腺感染。外部麦粒肿在眼睑外侧形成，可以看到是红色的小肿块。内部麦粒肿是眼睑内侧的睑板腺皮脂腺感染。它们还会在眼睑下方形成一个红色肿块，外部仅可见泛红和肿胀。麦粒肿与麦粒肿相似，但往往体积更小，疼痛更严重，而且通常不会产生持久的损伤。它们含有水和脓液，如果麦粒肿被用力弄破，细菌就会扩散。麦粒肿的特点是急性发作且持续时间通常较短（7-10 天，未经治疗），而霰粒肿是慢性的，通常不经干预无法解决。麦粒肿通常由金黄色葡萄球菌引起。   物理治疗练习；操纵；和其他程序   切开和引流（I d）   眼科检查和评估（眼科检查）   非手术去除异物   培养伤口     红霉素 ， 红霉素眼药 ， 磺胺醋酸钠眼药 ， 庆大霉素眼药 ， 妥布霉素眼药 ， 妥布霉素（Tobi） ， 丁卡因（一触式） ， 地塞米松 - 妥布霉素眼药 ， 庆大霉素（庆大霉素）   四氢唑啉眼科   荧光素眼科  

{thinking steps}:
根据所提供的医疗背景和医生的陈述，我们可以进行以下步骤的分析和指导：

对照医疗背景与医生陈述：
医疗背景：病人患有麦粒肿，表现为眼睛疼痛、肿胀、发红、眼睑肿胀等症状。麦粒肿通常由金黄色葡萄球菌引起，治疗方法包括物理治疗、药物治疗（例如红霉素、庆大霉素等眼药水）以及在必要时进行切开和引流。
医生陈述：医生建议使用左氧氟沙星眼水一天六次。

评估是否存在差异：
在提到的治疗方法中，虽然没有特别提到左氧氟沙星眼水，但它是一种广谱抗生素，通常用于治疗由细菌引起的眼部感染，包括麦粒肿。
因此，医生建议的治疗方案似乎是合理的，没有明显的差异。

提供指导和鼓励：
指导：确认医生对于病情的评估是否全面，例如是否有进行眼科检查和评估，以确保治疗方案的适宜性。同时，提醒注意观察患者对左氧氟沙星眼水的反应，如有任何异常反应应立即调整治疗方案。
鼓励：您选择的治疗方案是恰当的，左氧氟沙星眼水能有效针对麦粒肿的病原体。继续观察病人的反应和恢复情况，并根据需要调整治疗频率或药物种类。

{Output}:
医生，您对于麦粒肿患者使用左氧氟沙星眼水一天六次的建议是恰当的。这显示了您对疾病的专业理解和治疗的精准把握。对于麦粒肿，确保患者严格遵守用药频率和疗程是非常重要的，以促进快速恢复。同时，提醒患者注意个人卫生，避免用手触摸眼部，这样可以减少感染的风险，并防止病情恶化。

You will be given {Input}. Follow the above examples to provide your {thinking steps} and {Output} in Chinese.
{Input}:
"""


exit()

















