import pickle
import numpy as np




# slot = np.load('../generated_coach/old/slot_generated.npy', allow_pickle=True).item()


with open('gcot_gen.csv', 'rb') as f:
    v = pickle.load(f)

print(v)









