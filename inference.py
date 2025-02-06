# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import gradio as gr
import random
import torch
from transformers import AutoTokenizer
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.utils.config_utils import generate_dataset_config, get_dataloader_kwargs, update_config
from accelerate.utils import is_xpu_available
from llama_recipes.utils.memory_utils import MemoryTrace
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from prettytable import PrettyTable
import numpy as np
import json
import re

def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def generate_results(input_ids, model, tokenizer):
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        attention_mask=attention_mask,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print("-"*80)
    print("\nOutput:\n")
    print(tokenizer.decode(response, skip_special_tokens=True))
    print("-"*80)

def evaluation(model, dataset_raw, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, save_dir):
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    eval_preds = []
    results_saving = []
    total_eval_steps = 0
    # ret_table.float_format = ".4"
    # pattern = r'How would someone from (.*) answer the following question:'
    if "zh" in save_dir:
        pattern = r'来自(.*)的人会如何回答以下问题：\n\n'
    else:
        pattern = r'How would someone from (.*) answer the following question:'
    with MemoryTrace() as memtrace:
        all_jsd_scores, all_emd_scores = [], []
        try:
            for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
                torch.cuda.empty_cache()
                # stop when the maximum number of eval steps is reached
                if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                    if not train_config.enable_fsdp or local_rank==0:
                        print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                    break
                for key in batch.keys():
                    try:
                        if key == "labels":
                            continue
                        if key == "target_dist":
                            continue
                        if key == "input_length":
                            continue
                        # breakpoint()
                        batch[key] = torch.tensor(batch[key])
                        if train_config.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    except Exception as e:
                        print(f"Error in moving {key} to device: {e}")
                # Ensure no gradients are computed for this scope to save memory
                with torch.no_grad():
                    target_dist = batch.pop("target_dist")
                    input_length = batch.pop("input_length")
                    if 'labels' not in batch:
                        batch['labels'] = None
                    # breakpoint()
                    outputs = model(**batch)
                    # breakpoint()
                    # generate_results(batch["input_ids"], model, tokenizer)
                    for i, (input_len, target) in enumerate(zip(input_length, target_dist)):
                        # 获取第一 token 的 logit
                        cur_data = dataset_raw[total_eval_steps*train_config.val_batch_size+i]
                        cur_culture = re.search(pattern, cur_data['instruction']).group(1)
                        first_tok_logit = outputs.logits[i, input_len - 1]
                        # 分割 target
                        valid_mask = target != -1
                        target = torch.masked_select(target, valid_mask)
                        split_index = target.size(-1) // 2
                        golden_dist = target[:split_index].tolist()
                        golden_option_id = target[split_index:].long().tolist()
                        # 计算预测分布
                        pred_dist = torch.gather(first_tok_logit, dim=-1, index=torch.tensor(golden_option_id).to(first_tok_logit.device))
                        jsd_score = 1-distance.jensenshannon(golden_dist, torch.softmax(torch.tensor(pred_dist), dim=0).cpu())
                        emd_score = wasserstein_distance(torch.tensor(golden_dist)/100, torch.softmax(torch.tensor(pred_dist), dim=0).cpu())
                        if np.isnan(jsd_score):
                            print(f"JSD is nan, golden_dist: {golden_dist}, pred_dist: {pred_dist}")
                            continue
                        if np.isnan(emd_score):
                            print(f"EMD is nan, golden_dist: {golden_dist}, pred_dist: {pred_dist}")
                            continue
                        all_jsd_scores.append(jsd_score)
                        all_emd_scores.append(emd_score)
                        results_saving.append({"id": cur_data["id"], "culture": cur_culture,
                                            "1-jsd": jsd_score, "emd": emd_score, "pred_dist": pred_dist.tolist(),
                                            "golden_dist": golden_dist, 
                                            "response_text": tokenizer.decode(torch.argmax(outputs.logits[i], -1)[input_len-1:], skip_special_tokens=True),
                                            "golden_option_id": tokenizer.decode(golden_option_id),
                                            "data_type": cur_data["data_type"]})
                        # breakpoint()
                # # # # for debug
                # if len(results_saving) > 30:
                #     break
                total_eval_steps += 1
        except Exception as e:
            print(f"Error in evaluation: {e}")
    # save results_saving to json
    with open(save_dir + ".json", "w") as f:
        json.dump(results_saving, f, indent=4)
    print(f"Average JSD: {sum(all_jsd_scores) / len(all_jsd_scores)}")
    print(f"Average EMD: {sum(all_emd_scores) / len(all_emd_scores)}")
    print("saved results to ", save_dir + ".json")


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    test_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_dir: str = "output", #The directory to save the output files
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_infer = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="infer",
    )
    dataset_raw = read_json_file(dataset_config.data_path+dataset_config.infer_split+".json")
    val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_infer, tokenizer, "val")
    # breakpoint()
    test_dataloader = torch.utils.data.DataLoader(
        dataset_infer,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **val_dl_kwargs)
    print(f"--> Inference Set Length = {len(dataset_infer)}")
    print(f"--> Saving results to {save_dir}")
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        print("-"*20, "Loading PEFT model", "-"*20)
        model = load_peft_model(model, peft_model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    wandb_run = None
    
    evaluation(model,dataset_raw, train_config, test_dataloader, local_rank, tokenizer, wandb_run, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
