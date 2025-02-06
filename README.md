# Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations

This repository contains the code and resources for the NAACL 2025 main paper titled **"Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations"**. The project focuses on adapting large language models (LLMs) to simulate survey response distributions across diverse global populations.


## Introduction
This project explores the use of large language models (LLMs) to simulate survey responses for global populations. By fine-tuning LLMs on survey data, we aim to generate realistic response distributions that reflect the diversity of global opinions. 

---

## Installation
To set up the environment, clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation
Before running the experiments, you need to prepare the dataset. Run the following script to download and preprocess the data:

```bash
sh prepare_data.sh
```

or you can also directly use our processed data: [Download Dataset](https://drive.google.com/drive/folders/1pijtrk5yW7-KnkLXl9TlNu1J3RAHFMbO?usp=sharing).

---

## Zero-shot Evaluation
To evaluate the model in a zero-shot setting, use the following script:

```bash
sh infer_slurm.sh
```


## Fine-tuning
To fine-tune the model on the survey dataset, run the following script:

```bash
sh train_slurm.sh
```

## Evaluation
After fine-tuning, you can evaluate the model's performance using the following script (config INFER_MODE as sft, and CKPT_PATH):

```bash
sh infer_slurm.sh
```

<!-- ## Citation
If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{yourcitationkey,
  title={Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2025}
}
``` -->

---

## Contact
For any questions or issues, please open an issue on this repository or contact yongcao2018@gmail.com.