import pickle
import glob
import numpy as np


prefix = 'translation_20_match/translated_conversation_*'


def get_dic(file_prefix):
    file_paths = glob.glob(file_prefix)

    sorted_file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1] != '' else -float('inf'))

    ### dictionary of whole data
    translation_dictionary = {}
    for path in sorted_file_paths:
        with open(path, 'rb') as f:
            d = pickle.load(f)
            translation_dictionary.update(d)

    return translation_dictionary


translation = get_dic(prefix)
keylist = translation.keys()
# print(keylist)
print(len(translation))

translated_conversation = ''
for key, value in translation.items():
    translated_conversation = translated_conversation + key + '\n'
    translated_conversation += value + '\n'


# with open("translated_conversation.txt", "w", encoding='utf-8') as file:
#     file.write(translated_conversation)









