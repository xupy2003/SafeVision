import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import json

path = './InternVL2_5-8B'

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

print("token length:", len(tokenizer))

tokenizer.add_tokens([
    "<|Safe|>",
    "<|NSFW|>",
    "<|REASON|>"
])

print("token length:", len(tokenizer))

old_embedding_weights = model.language_model.model.tok_embeddings.weight.data
new_tok_embeddings = torch.nn.Embedding(len(tokenizer), old_embedding_weights.size(1), padding_idx=2)
new_tok_embeddings.weight.data[:old_embedding_weights.size(0)] = old_embedding_weights
model.language_model.model.tok_embeddings = new_tok_embeddings

old_output_layer = model.language_model.output
new_output_layer = torch.nn.Linear(in_features=old_output_layer.in_features, out_features=len(tokenizer), bias=False)
new_output_layer.weight.data[:old_output_layer.out_features] = old_output_layer.weight.data
model.language_model.output = new_output_layer

save_path = '/scratch/xpy/image_moderation/InternVL/internvl_chat/pretrained/2b_nsfw_specific'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to {save_path}")

config_path = os.path.join(save_path, "config.json")
with open(config_path, "r") as file:
    config_data = json.load(file)
config_data["vocab_size"] = len(tokenizer)
with open(config_path, "w") as file:
    json.dump(config_data, file, indent=4)

print(f"Config file updated with new vocab_size: {config_data['vocab_size']}")
