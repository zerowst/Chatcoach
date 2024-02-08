import json
import pickle
import numpy as np


def main():
    with open('all_cases', 'rb') as f:
        all_cases = pickle.load(f)

    with open('lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    data_json = {}

    for key, all_values in all_cases.items():
        data_json[key] = {}
        data_json[key]['medical'] = all_values['medical']

        data_con = []
        for i in range(len(all_values['con']['d'])):
            data_con.append(all_values['con']['d'][i])
            data_con.append(all_values['con']['c'][i])
            data_con.append(all_values['con']['p'][i])

        data_json[key]['conversation'] = data_con

    # testing_set = json.dumps(data_json, ensure_ascii=False)
    testing_set = {str(i+1): value for i, value in enumerate(data_json.values())}
    with open('testing.json', 'w', encoding='utf-8') as f:
        json.dump(testing_set, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()






























