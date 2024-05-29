import numpy as np
import csv
import os
import pickle
import re
from docx import Document
import random

###-----------------------------------------------------------------------------------------
# Measure the number of human labels in each case


###-----------------------------------------------------------------------------------------


file_human = 'annotation_data/annotation_human.docx'
human_file = Document(file_human).paragraphs
human_lines = [para.text for para in human_file if para.text.strip()]


human_label = {}
machine_label = []
n_labels = {}
n_case = 0
wrong_case = []
for i, line in enumerate(human_lines):
    if line.startswith('Case:'):

        n_case += 1
        human_label[n_case] = []
        n_labels[n_case] = 0


    if line.startswith('人工标注'):
        n_labels[n_case] += 1
        detect_correct = re.split('人工标注：|修改为', line)
        ### try to split line into three parts
        if len(detect_correct) == 3:
            human_label[n_case].append([detect_correct[1], detect_correct[2]])

            ### line has less than 3 elements
        elif len(detect_correct) < 3:

            ### human label is empty, move to next line. Do the same check and split
            if human_lines[i + 1].startswith('错误术语'):

                detect_correct = re.split('错误术语：|修改为', human_lines[i + 1])
                if len(detect_correct) == 3:
                    human_label[n_case].append([detect_correct[1], detect_correct[2]])
                else:
                    print('wrong case of next line')
                    print(n_case)
                    print(detect_correct)


            else:
                print('wrong case')
                print(n_case)
                wrong_case.append(n_case)
                print(line)
                print(human_lines[i + 1])

        else:
            print('wrong case')
            print(n_case)
            wrong_case.append(n_case)
            print(line)
            print(human_lines[i + 1])
            print('long line', line)



n_labels = np.array(list(n_labels.values()))
zero_case = np.where(n_labels==0)[0]
zero_case = zero_case+1
np.save('zero_case', zero_case)
wrong_case = np.unique(wrong_case)
correct_keys = []
for k in range(1, 501):
    if k not in zero_case and k not in wrong_case:
        correct_keys.append(k)



detection_label = {}
correction_label = {}

symbols = r'[：。，、（）]|\b错误术语\b'
for label in correct_keys:
    detection_label[label] = []
    correction_label[label] = []
    for pair in human_label[label]:
        det_label = re.sub(symbols, '', pair[0]).replace(' ', '')
        cor_label = re.sub(symbols, '', pair[1]).replace(' ', '')
        detection_label[label].append(det_label)
        correction_label[label].append(cor_label)

print(len(detection_label))


with open('human_detection_all', 'wb') as f:
    pickle.dump(detection_label, f)

with open('human_correction_all', 'wb') as f:
    pickle.dump(correction_label, f)






