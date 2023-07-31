import argparse
import json
import os
import gc
import torch
import peft
from peft import (PeftModel, LoraConfig, LoraModel, get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field

BASE_MODEL_PATH = 'FlagAlpha/Llama2-Chinese-7b-Chat'
LORA_MODEL_PATH = '../train/sft/output_path/checkpoint-300'

lora_r: Optional[int] = field(default=8)
lora_alpha: Optional[int] = field(default=32)
target_modules: Optional[str] = field(
    default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
    metadata={
        "help": "List of module names or regex expression of the module names to replace with Lora."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
    },
)

load_in_bits: Optional[int] = field(default=4)

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    # target_modules=["query_key_value"],
    # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    target_modules=target_modules,
    fan_in_fan_out=False,
    lora_dropout=0.05,
    inference_mode=False,
    bias="none",
    task_type="CAUSAL_LM",
)





tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)

model = LlamaForCausalLM.from_pretrained(BASE_MODEL_PATH,device_map='auto', use_auth_token=True)

# lora_model = PeftModel.from_pretrained(model, LORA_MODEL_PATH, adapter_name='lora1')
# model = get_peft_model(model, lora_config)


embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
if load_in_bits == 8:
    model = prepare_model_for_int8_training(model)
elif load_in_bits == 4:
    model = prepare_model_for_kbit_training(model)






