import pickle
import numpy as np

# --------------------------------------------------------------------------------------
# get relative position of current selected idx to origin human label idx


# ------------------------------------------------------

with open('../../../human_label_data/lingual_pos', 'rb') as f:
    new_pos = pickle.load(f)


with open('../lingual_pos', 'rb') as f:
    old_pos = pickle.load(f)

relative_pos = {}
for case, new_positions in new_pos.items():
    relative_pos[case] = []
    for i, new_p in enumerate(new_positions):
        old_positions = old_pos[int(case)]
        relative_pos[case].append(old_positions.index(new_p))


with open('relative_pos', 'wb') as f:
    pickle.dump(relative_pos, f)

case = new_pos.keys()
print(case)














