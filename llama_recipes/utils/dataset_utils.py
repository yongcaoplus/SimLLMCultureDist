# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

import torch

# from llama_recipes.datasets import (
#     get_grammar_dataset,
#     get_alpaca_dataset,
#     get_samsum_dataset,
# )


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if dataset_config.model_type == "base":
        module_path, func_name = dataset_config.file, "BasicDataset"
    elif dataset_config.model_type == "instruct":
        module_path, func_name = dataset_config.file, "InstructionDataset"
    elif dataset_config.model_type == "qwen":
        module_path, func_name = dataset_config.file, "QwenDataset"
    elif dataset_config.model_type == "GLM-4":
        module_path, func_name = dataset_config.file, "GLMDataset"
    elif dataset_config.model_type == "baichuan":
        module_path, func_name = dataset_config.file, "BaichuanDataset"
    elif dataset_config.model_type == "vicuna-7b" or dataset_config.model_type == "vicuna-13b":
        module_path, func_name = dataset_config.file, "VicunaDataset"
    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


# "alpaca_dataset": partial(get_alpaca_dataset),
# "grammar_dataset": get_grammar_dataset,
# "samsum_dataset": get_samsum_dataset,

DATASET_PREPROC = {
    "custom_dataset": get_custom_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    # def get_split():
    #     return (
    #         dataset_config.train_split
    #         if split == "train"
    #         else dataset_config.test_split
    #     )
    def get_split():
        if split == "train":
            return dataset_config.train_split
        elif split == "infer":
            return dataset_config.infer_split
        else:
            return dataset_config.test_split
        # return (
        #     dataset_config.train_split
        #     if split == "train"
        #     else dataset_config.test_split
        # )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
