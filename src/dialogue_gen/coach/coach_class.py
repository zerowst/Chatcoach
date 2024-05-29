import openai
import numpy as np
import pickle
import random
import pandas as pd
import Dialogue_gen.combine_dataset as combine
import Embedding.dialogue_embed as embed
import os




class Coach():
    def __init__(self):
        # self.api_key = api_key
        self.directory_path = os.path.dirname(os.path.abspath(__file__))

        self.dic_all, self.label, self.match_distance_key, self.generated_dic, self.dialogue_dic = self.load_conversation_files()
        self.disease_all, self.disease_name, self.disease_embedding = self.load_disease_files()

    def load_conversation_files(self):

        dialogue_file = os.path.join(self.directory_path,  '../../dia_doc')
        doctor_file = os.path.join(self.directory_path, '../../doc_sen_doc')
        gen_doc_file = os.path.join(self.directory_path, '../misuse_20/')

        file_prefix = os.path.join(self.directory_path, '../misuse_20/save_dic_misuse_*')

        match_key_file = os.path.join(self.directory_path, 'key_extraction/match_distance_key_20.npy')

        dictionary_all_file = os.path.join(self.directory_path, '../../parsed_2020t')

        ###origin conversation

        ### whole conversation
        with open(dialogue_file, 'rb') as f:
            dialogue_complete = pickle.load(f)

        with open(doctor_file, 'rb') as f:
            doc_dictionary = pickle.load(f)

        with open(dictionary_all_file, 'rb') as f:
            dic_all = pickle.load(f)
        label = ['医生：', '病人：']

        match_distance_key = np.load(match_key_file)

        gen_dic = self.generated_sen_process(file_prefix, doc_dictionary)

        dialogue_dic = {}
        for key in dialogue_complete.keys():
            if len(dialogue_complete[key]) != 0:
                dialogue_dic[key] = dialogue_complete[key]

        return dic_all, label, match_distance_key, gen_dic, dialogue_dic

    def load_disease_files(self):
        disease_all_file = os.path.join(self.directory_path, '../../Embedding/disease_all')
        disease_name_file = os.path.join(self.directory_path, '../../Embedding/disease_name')
        disease_embedding_file = os.path.join(self.directory_path, '../../Embedding/disease_embedding.npy')

        with open(disease_all_file, 'rb') as f:
            disease_all = pickle.load(f)

        with open(disease_name_file, 'rb') as f:
            disease_name = pickle.load(f)

        disease_embedding = np.load(disease_embedding_file)
        return disease_all, disease_name, disease_embedding

    @staticmethod
    def language_level(self, mean=0.6, std=1):
        ### correct prob
        data = np.random.normal(mean, std, 1000) / 6 + 0.5
        level = np.random.choice(data)
        level = np.clip(level, 0, 1)
        return level

    @staticmethod
    def random_selection(self, prob):
        prob_w = [prob, 1 - prob]
        ###
        return np.random.choice([1, 0], p=prob_w)

    def gpt_coach(self, medical_context, conversation, regen=1, max_regen=5, retry=1, max_retry=5):
        prompt = """Based on the provided <medical context>:\n""" + \
                 """<medical_context>:\n""" + medical_context + '\n' + \
                 """Here is given conversation:\n""" + """<conversation>:\n""" + \
                 conversation + '\n' + \
                 """In this simulated dialogue_gen, you are a language coach assisting a doctor in providing accurate medical advice. You only cae about doctor's sentences.
                 Your task is to identify any discrepancies between the doctor's dialogue_gen and the correct responses provided for comparison. 
                 Your role is to guide the doctor in making more precise and accurate statements, taking into consideration the given medical context.
                 When identifying errors in the doctor's statements regarding symptoms, you should say:
                 "Doctor, there may be some misunderstandings about the description of symptoms. The term '<incorrect symptom words>' seems inappropriate in the context of '<current disease>'. You might need to consider using the following terminology: '<correct symptoms>'."
                 When pointing out errors related to disease or medication names, use this pattern:
                 "Doctor, the medical term you used seems to be incorrect. The word '<incorrect words>' is not accurate. The correct term to use would be '<correct words>'."
                 Please remember that '<incorrect symptom words>' and '<incorrect words>' refer to the words used by the doctor which do not match the terms provided in the correct response. '<current disease>' and '<correct symptoms>' are derived from the medical context provided.
                 For any other errors, you are not required to provide feedback.
                 Please note that you should interact directly with the doctor. Your task is to revise the dialogue_gen and provide the coaching directly to the doctor. For a more natural dialogue_gen, please remove references to the correct responses.
     
                 The conversation flow in the generated dialogue_gen should be as follows:
                 1 Doctor: Initial medical statement
                 2 Coach: Correction, if needed
                 3 Patient: Response
                 4 Repeat from 1 to 3
     
                 The conversation form should be:
                 医生：
                 教练：
                 病人：
                 ...
                 Until the end of the conversation
     
                 Use the given information (including difference between the doctor's dialogue_gen and the correct responses as well as the provided medical context) to add content for the coach and rewrite the dialogue_gen if necessary. The dialogue_gen includes three participants: the doctor, the patient, and the coach.
                 The coach should speak Chinese."""
        try:
            coach_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.85,
                # api_key=self.api_key
            )
            response = coach_response['choices'][0]['message']['content']
            print('chatgpt\n')

            if self.check_response(response):
                print('Generate successfully')
                return response
            elif regen < max_regen:
                print('regeneration\n')
                return self.gpt_coach(medical_context, conversation, regen=regen + 1)
            else:
                print("Regeneration has reached limit")
                return ''

        except openai.OpenAIError as e:
            if retry < max_retry:
                print(f'GPT error: {e}, retry={retry}')
                return self.gpt_coach(medical_context, conversation, retry=retry + 1)
            else:
                print(f'GPT error retrying has reached limit.')
                exit(1)

    def construct(self, dia_dic, gen_dic, correct_key):
        for key in correct_key[:100]:
            length = len(dia_dic[key])
            if length == 2:
                dia_dic[key][1] = gen_dic[key][1]
            else:
                idx = 0
                for i in range(1, length, 4):
                    dia_dic[key][i] = gen_dic[key][idx] + '\n'
                    # print(gen_dic[key][idx])
                    idx += 1

        return dia_dic

    def construct_conversation(self, ori_dia, gen_dia):
        new_dia = []
        length = len(ori_dia)
        prob = self.language_level()
        if length == 2:

            ori_dia[1] = gen_dia[0]
        else:
            idx = 0
            for i in range(0, length, 4):
                correct = self.random_selection(prob)
                # print(correct)
                new_sen = ori_dia[i + 1] if correct else gen_dia[idx] + '\n'

                new_dia.append(ori_dia[0])
                new_dia.append(new_sen)
                new_dia.append("正确的句子：\n")
                new_dia.append(ori_dia[i + 1])
                new_dia.append("教练：\n")
                new_dia.append("<>\n")
                new_dia = new_dia + ori_dia[i + 2:i + 4] + ['\n']

                # print(gen_dic[key][idx])
                idx += 1
                # print(new_dia)

        return new_dia

    def check_response(self, response):
        # detect '<>'
        if '<' in response:
            # print('< is in response')
            return 0

        response_list = response.splitlines()
        # detect coach
        flag = 0
        for res in response_list:
            if res.startswith('教练') or res.startswith('Coach') or res.startswith('coach'):
                flag += 1
                break
        flag_d = 0
        for res in response_list:
            if res.startswith('医生') or res.startswith('Doctor') or res.startswith('doctor'):
                flag_d += 1
                break
        flag_p = 0
        for res in response_list:
            if res.startswith('病人') or res.startswith('Patient') or res.startswith('patient'):
                flag_p += 1
                break

        # print(flag, flag_d, flag_p)
        if flag and flag_d and flag_p:
            return 1
        else:
            return 0

    def medical_context(self, key, dic, disease, disease_name):

        diag = dic[key]['Diagnosis and suggestions'][1]
        if diag in disease_name.values:
            # print(diag)
            med_idx = disease_name.values.tolist().index(diag)
            # print(med_idx)
            context = disease[med_idx + 1]
            # print(context)

            return context
        else:
            print('Wrong key')
            return ''

    def generated_sen_process(self, file_prefix, doc_dictionary):
        generated_dictionary, generated_key = combine.get_dic_key(file_prefix)
        correct_key, incorrect_key, ratio_wrong = combine.correct_rate(generated_key, doc_dictionary,
                                                                       generated_dictionary)

        ### alignment length part

        gen_correct_dictionary = {}
        for key in correct_key:
            gen_correct_dictionary[key] = combine.get_response(generated_dictionary[key])

        ### origin conversation with one doctor sentence
        one_line_key = combine.analysis_wrong_list(incorrect_key, doc_dictionary, 2)
        one_line_key_dict = {}
        for key in one_line_key:
            one_line_key_dict[key] = combine.get_response(generated_dictionary[key][:2])

        ### combine alignment part and one_line part
        gen_correct_dictionary.update(one_line_key_dict)
        return gen_correct_dictionary

    def run(self, save_prefix, start_idx, end_idx):

        i = start_idx
        e = end_idx
        save_coach_sen = {}
        # save_filename = 'coach_distance_20_' + str(i)
        save_filename = save_prefix + str(i)
        correct_match_key = self.match_distance_key[i:e]

        for key in correct_match_key:
            new_conv = self.construct_conversation(self.dialogue_dic[key], self.generated_dic[key])
            conver_text = embed.combine_text(self.dic_all, str(key))
            dia_embed = embed.get_embedding(conver_text)
            max_index = embed.calculating_distance(dia_embed, self.disease_embedding)[-1]
            med_context = self.disease_all[max_index]

            new_dialogue = ''
            for dia in new_conv:
                new_dialogue += dia
            response = self.gpt_coach(med_context, new_dialogue)
            save_coach_sen[key] = response

            if i % 10 == 0:
                with open(save_filename, 'wb') as f:
                    pickle.dump(save_coach_sen, f)

                print("Saved number: ", i)
                print('\n')

                print("\nModified: \n", response)
                print('\n')

            i += 1

# i = 0
# e = 1000
# save_coach_sen = {}
# save_pre = 'coach_distance_20_'
#
# coach_1 = Coach()
# coach_1.run(save_pre, i, e)

