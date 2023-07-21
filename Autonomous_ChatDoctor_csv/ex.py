import numpy as np
import pandas as pd
import json
import openai

# with open("../datafile/HealthCareMagic-100k.json") as f:
#     data = json.load(f)
#
# # print(data[7320].keys())
# print(len(data))
# print(data[0].keys())
#
# print(data[0])
from transformers import cached_path

cache_dir = cached_path("", force_download=False)
print(cache_dir)


a = 1
b = 2

print("a: {}, b: {}".format(a, b))










