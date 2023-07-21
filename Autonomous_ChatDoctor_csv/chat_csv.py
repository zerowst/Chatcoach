import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import pipeline,LlamaTokenizer,LlamaForCausalLM
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
import numpy as np

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from csv_reader import csv_prompter

def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading " + model_name + "...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        # device_map=device_map,
        # device_map="auto",
        torch_dtype=torch.float16,
        # max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        # load_in_8bit=eight_bit,
        # from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

def go():
    invitation = "ChatDoctor: "
    human_invitation = "Patient: "


    # # input
    # msg = iclinic[0]['input']
    # print("")
    #
    # response = csv_prompter(generator,tokenizer,msg)
    #
    # print("")
    # print(invitation + response)
    # print("")
    for i in range(500):
        msg = iclinic[i]['input']

        print("Number of sample: ", i+1)
        print(msg)
        response = csv_prompter(generator, tokenizer, msg)
        print(" ")
        print(invitation + response)
        print(" ")
        answer.append(response)
        if i%10 == 0:
            np.save('iclinic_answer_time', answer)


if __name__ == '__main__':
    load_model("../pretrained/")

    First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
    print(First_chat)

    with open("../datafile/iCliniq.json") as f:
        iclinic = json.load(f)

    sample = len(iclinic)
    answer = []

    go()

