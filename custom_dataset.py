# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset
import random 

fix_prompt = " If had to select one of the options, my answer would be ("

class BasicDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: base ", "-"*10)
        self.max_sent_len = dataset_config.max_sent_len
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer
        self.begin_of_text_id = self.tokenizer.get_vocab()["<|begin_of_text|>"]
        self.start_header_id = self.tokenizer.get_vocab()["<|start_header_id|>"]
        self.end_header_id = self.tokenizer.get_vocab()["<|end_header_id|>"]
        self.eot_id = self.tokenizer.get_vocab()["<|eot_id|>"]
        self.nl_tokens = [self.tokenizer('\n').input_ids[1]]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"])
        input_id = self.tokenizer.encode(input_text+fix_prompt, padding="max_length", max_length=self.max_sent_len, truncation=True)
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into 20
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        input_length = len(self.tokenizer.encode(input_text+fix_prompt))
        # labels[~label_mask] = IGNORE_INDEX
        # breakpoint()
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }

class VicunaDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: base ", "-"*10)
        self.max_sent_len = dataset_config.max_sent_len
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"])
        input_id = self.tokenizer.encode(input_text+fix_prompt, padding="max_length", max_length=self.max_sent_len, truncation=True)
        # breakpoint()
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into 20
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        input_length = len(self.tokenizer.encode(input_text+fix_prompt))
        # labels[~label_mask] = IGNORE_INDEX
        # breakpoint()
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: instruction ", "-"*10)
        self.max_sent_len = dataset_config.max_sent_len
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer
        self.begin_of_text_id = self.tokenizer.get_vocab()["<|begin_of_text|>"]
        self.start_header_id = self.tokenizer.get_vocab()["<|start_header_id|>"]
        self.end_header_id = self.tokenizer.get_vocab()["<|end_header_id|>"]
        self.eot_id = self.tokenizer.get_vocab()["<|eot_id|>"]
        self.nl_tokens = [self.tokenizer('\n').input_ids[1]]
        self._system = [self.tokenizer('system').input_ids[1]]
        self._user = [self.tokenizer('user').input_ids[1]]
        self._assistant = [self.tokenizer('assistant').input_ids[1]]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"])
        input_id = [self.begin_of_text_id] + [self.start_header_id] + self._user + [self.end_header_id] \
                    + self.tokenizer(input_text).input_ids[1:] + \
                    [self.eot_id] + [self.start_header_id] + self._assistant + [self.end_header_id] + \
                    self.tokenizer(fix_prompt).input_ids[1:]
        input_length = len(input_id)
        # padding and truncating
        if len(input_id) < self.max_sent_len:
            input_id = input_id + [self.eot_id] * (self.max_sent_len - len(input_id))
        elif len(input_id) > self.max_sent_len:
            input_id = input_id[:self.max_sent_len]
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into 20
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }

class BaichuanDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: instruction ", "-"*10)
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer
        self.max_sent_len = dataset_config.max_sent_len
        self.user_tokens=[195]
        self.assistant_tokens=[196]
        self.ignore_index = -100
        self.eos_tok = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        input_id = []
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"])
        value_ids = self.tokenizer.encode(input_text)
        input_id += self.user_tokens + value_ids + self.assistant_tokens + self.tokenizer.encode(fix_prompt)
        input_length = len(input_id)
        # padding and truncating
        if len(input_id) < self.max_sent_len:
            input_id = input_id + [self.tokenizer.eos_token_id] * (self.max_sent_len - len(input_id))
        elif len(input_id) > self.max_sent_len:
            input_id = input_id[:self.max_sent_len]
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into at most 30 options
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }


class GLMDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: instruction ", "-"*10)
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer
        self.max_sent_len = dataset_config.max_sent_len
        self.user_tokens=[195]
        self.assistant_tokens=[196]
        self.eos_tok = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        input_id = []
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"])
        # breakpoint()
        # input_id = self.tokenizer.apply_chat_template([{"role": "user", "content":"1+1="}], add_generation_prompt=True,tokenize=True,return_tensors="pt",return_dict=True)
        value_ids = self.tokenizer.encode(input_text)
        input_id += self.user_tokens + value_ids + self.assistant_tokens + self.tokenizer.encode(fix_prompt)
        input_length = len(input_id)
        # padding and truncating
        if len(input_id) < self.max_sent_len:
            input_id = input_id + [self.eos_tok] * (self.max_sent_len - len(input_id))
        elif len(input_id) > self.max_sent_len:
            input_id = input_id[:self.max_sent_len]
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into at most 30 options
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }

class QwenDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        print("Dataset config: ", dataset_config)
        print("partition: ", partition)
        print("-"*10, "Dataset type: instruction ", "-"*10)
        self.ann = json.load(open(dataset_config.data_path + partition + ".json"))
        self.tokenizer = tokenizer
        self.max_sent_len = dataset_config.max_sent_len
        self.eos_tok = self.tokenizer.eos_token
        self.system_message = "You are a helpful assistant."
        # breakpoint()
        self.roles = {"user": "<｜begin▁of▁sentence｜>user", "assistant": "<｜begin▁of▁sentence｜>assistant"} 
        self.im_start = self.tokenizer.get_vocab()["<｜begin▁of▁sentence｜>"]
        self.im_end = self.tokenizer.get_vocab()["<｜end▁of▁sentence｜>"]
        # self.roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"} 
        # self.im_start = self.tokenizer.get_vocab()["<|im_start|>"]
        # self.im_end = self.tokenizer.get_vocab()["<|im_end|>"]
        self.nl_tokens = tokenizer('\n').input_ids
        self._system = tokenizer('system').input_ids + self.nl_tokens
        self._user = tokenizer('user').input_ids + self.nl_tokens
        self._assistant = tokenizer('assistant').input_ids + self.nl_tokens

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        input_id = []
        ann = self.ann[index]
        input_text = ann["instruction"] + ann["input"] + ", ".join(ann["options"]) + fix_prompt
        system = [self.im_start] + self._system + self.tokenizer(self.system_message).input_ids + [self.im_end] + self.nl_tokens
        input_id += system
        _input_id = self.tokenizer(self.roles['user']).input_ids + self.nl_tokens + \
                    self.tokenizer(input_text).input_ids + [self.im_end] + self.nl_tokens
        input_id += _input_id
        _input_id = self.tokenizer(self.roles['assistant']).input_ids + self.nl_tokens + \
                    self.tokenizer(fix_prompt).input_ids
        input_id += _input_id    
        input_length = len(input_id)
        # padding and truncating
        if len(input_id) < self.max_sent_len:
            input_id = input_id + self.tokenizer.encode(self.eos_tok) * (self.max_sent_len - len(input_id))
        elif len(input_id) > self.max_sent_len:
            input_id = input_id[:self.max_sent_len]
        example = torch.tensor(
            input_id, dtype=torch.int64
        )
        target = [ann["options_dist"][item[1]] if item[1] in ann["options_dist"] else 0 for item in ann["options"]]
        target_id = [self.tokenizer.encode(option[1:])[1] for option in ann['options']]
        targets = target+target_id
        # padding targets into at most 30 options
        targets += [-1] * (60 - len(targets))
        example_mask = example.ge(0)
        example[~example_mask] = 0
        return {
            "input_ids": example.tolist(),
            "attention_mask":example_mask.tolist(),
            "target_dist": targets,
            "input_length": input_length
        }