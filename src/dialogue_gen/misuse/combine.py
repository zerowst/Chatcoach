import numpy as np
import pickle
import pandas as pd
import os
import random

### loading    start with doctor
# order case
with open('misuse/misuse_20/save_dic_order', 'rb') as f:
    order_cases = pickle.load(f)

# misuse
with open('misuse/misuse_20/save_dic_misuse', 'rb') as f:
    misuse_cases = pickle.load(f)

# patient
with open('../pat_sen_doc', 'rb') as f:
    patient = pickle.load(f)

# doctor
with open('../doc_sen_doc', 'rb') as f:
    doctor = pickle.load(f)

keys = ['4', '6', '11', '18', '24', '27', '31']
label = ['医生：', '病人：']

def random_selection():
    idx = [0, 1, 2]  # origin misuse order
    prob = [0.4, 0.3, 0.3]

    return random.choice(idx, prob, 1)[0]


for key in keys:
    line_m = []
    lines = misuse_cases[key].splitlines()
    for line in lines:
        if line.strip():
            line_m.append(line)
    print(len(line_m))


print('\n\n')

for key in keys:
    line_m = []
    lines = order_cases[key].splitlines()
    for line in lines:
        if line.strip():
            line_m.append(line)
    print(len(line_m))

print('\n\n')

for key in keys:
    print(len(doctor[key]))


def get_response(response):
    # strip()
    response_list = []
    response_lines = response.splitlines()
    for response_line in response_lines:
        if not response_line.strip():
            response_lines.pop(response_line)

    # print(response_lines)
    len_response = len(response_lines)

    # find label and separate sentence
    label_idx = np.where(np.array(response_lines) == label[0])[0]
    label_idx = np.append(label_idx, len_response)

    start = label_idx[0]
    for i in range(len(label_idx)-1):
        sen = ''
        end = label_idx[i+1]
        sublist = response_lines[start+1:end]
        sen = ''.join(sublist)

        if sen.strip():
            response_list.append(sen)
        start = end
    return response_list


for key in keys:
    misuse_cases[key] = get_response(misuse_cases[key])
    print(misuse_cases[key])
    order_cases[key] = get_response(order_cases[key])
    print(order_cases[key])
    # doctor[key] = get_response(doctor[key])
    print('\n')


print("adjusted:\n\n")

for key in keys:
    print(key)

    print(len(misuse_cases[key]))
    print(len(order_cases[key]))

    print(len(doctor[key]))
    print('\n')



