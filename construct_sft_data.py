import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import argparse


def load_questions(dataset_path, lan="en"):
    data = json.load(open(dataset_path, "r"))
    probing_ques = []
    for i, item in enumerate(data[lan]):
        question = item['question']
        probing_ques.append(question+" \nHere are the options: \n")
    return data, probing_ques

def fill_prompt(culture):
    prompt = "How would someone from {Country} answer the following question:\n".format(Country=culture)
    return prompt

def format_query(prompt, questions):
    return [prompt + ques for ques in questions]


def plot_culture_dist(culture_id2name, sample_distribution):
    # sort by culture name
    sample_distribution = dict(sorted(sample_distribution.items(), key=lambda item: item[1]))
    # breakpoint()
    culture_names = [culture_id2name[culture] for culture in sample_distribution.keys()]
    culture_counts = list(sample_distribution.values())
    plt.figure(figsize=(20, 6))
    plt.bar(culture_names, culture_counts, color='#7bdafb')
    plt.xticks(rotation=90)
    # plt.title("Sample distribution of cultures")
    plt.xlabel("Country")
    plt.ylabel("Number of samples")
    # label with counts
    # for i in range(len(culture_counts)):
    #     plt.text(i, culture_counts[i]+50, str(culture_counts[i]), ha='center')
    plt.tight_layout()
    plt.show()
    plt.savefig("dataset/sample_distribution.png", dpi=300)


def get_wvs_golden_scores(file_path, culture_id2name):
    wvs_dimensions_range = {
            "social values, attitudes & stereotypes": [1, 45, 45],
            "societal well-being": [46, 56, 11],
            "social capital, trust and organizational membership": [57, 105, 49],
            "economic values": [106, 111, 6],
            "corruption": [112, 120, 9],
            "migration": [121, 130, 10],
            "security": [131, 151, 21],
            "post-materialist index": [152, 157, 6],
            "science & technology": [158, 163, 6],
            "religious values": [164, 175, 12],
            "ethical values & norms": [176, 198, 23],
            "political interest and political participation": [199, 234, 36],
            "political culture and political regimes": [235, 259, 24]
        }
    num2alpha = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 0: "X"}
    wvs_dimensions_index = {}
    for dimension, index_range in wvs_dimensions_range.items():
        wvs_dimensions_index[dimension] = [item for item in range(index_range[0], index_range[1] + 1)]
    data = pd.read_csv(file_path)
    culture_in_data = data["B_COUNTRY"].unique()
    # Counter 
    sample_distribution = Counter(data["B_COUNTRY"])
    # print("Sample distribution: ", sample_distribution)
    # plot distribution
    plot_culture_dist(culture_id2name, sample_distribution)
    # breakpoint()
    golden_wvs = {}
    for culture in culture_in_data:
        dimension_score = {}
        answer_number = []
        for ques_id in range(1, 260):
            answer_dist = data.loc[data["B_COUNTRY"] == culture, "Q" + str(ques_id)].tolist()
            answer_number.append(len(answer_dist))
            answer_dist = [x for x in answer_dist if 0 <= x <= 10]
            answer_dist = dict(Counter(answer_dist))
            total_count = sum(answer_dist.values())
            distribution_percentage = {num2alpha[key]: (value / total_count) * 100 for key, value in answer_dist.items()}
            dimension_score[ques_id] = distribution_percentage
        golden_wvs[culture_id2name[culture]] = dimension_score
        # print(answer_number)
        # breakpoint()
    return golden_wvs, [culture_id2name[item] for item in culture_in_data]

def form_answer(input_question, golden_scores, culture, ques_id):
    distribution_percentage = golden_scores[culture][ques_id]
    text_description = ""
    for option, percentage in distribution_percentage.items():
        text_description += f"{percentage:.2f}% of people select option {option}, "
    text_description = text_description[:-2]
    return text_description


def save_to_json(data, save_path, save_type):
    print("{}: {} saved to {}".format(save_type, len(data), save_path))
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def check_dist_in_options(input_question, golden_scores, cur_culture, cur_ques_id, options_text):
    answer_dist_key = golden_scores[cur_culture][cur_ques_id].keys()
    for item in answer_dist_key:
        if str(item) not in options_text:
            print("Key not in options: ", item, "  ", options_text)
            breakpoint()

def format_data(countries, questions, data_type, exclude_ques, all_question, golden_scores, original_ques):
    all_data = []
    # culture_name2id = {v: k for k, v in culture_id2name.items()}
    if "Northern Ireland" in countries:
        countries.remove("Northern Ireland")
    len_questions = len([item for item in range(questions[0], questions[1]) if item not in exclude_ques])
    # print("Processing {} questions for {} cultures, all: {}".format(len_questions, len(countries), len_questions*len(countries)))
    # print(" | ".join(countries))
    # print(questions)
    # print("-"*20)
    # print("Culture: {}, Questions: {}, Case Number: {}".format(len(countries), len_questions, len_questions*len(countries)))
    for cur_culture in tqdm(countries):
        for cur_ques_id in range(questions[0], questions[1]):
            if cur_ques_id in exclude_ques:
                continue
            instruction = fill_prompt(cur_culture)
            input_question = original_ques['en'][cur_ques_id-1]
            input_text = all_question[cur_ques_id-1]
            options = input_question['answers']
            if len(options) < len(golden_scores[cur_culture][cur_ques_id]):
                breakpoint()
            if len(golden_scores[cur_culture][cur_ques_id]) < 1:
                continue
            for item in options:
                if item[1] not in golden_scores[cur_culture][cur_ques_id]:
                    golden_scores[cur_culture][cur_ques_id][item[1]] = 0.0
            # output_text = form_answer(input_question, golden_scores, cur_culture_num, cur_ques_id)
            all_data.append({"id":input_question['id'], "instruction": instruction+"\n", "input": input_text, "options":options, "options_dist": golden_scores[cur_culture][cur_ques_id], "data_type": data_type})
            check_dist_in_options(input_question, golden_scores, cur_culture, cur_ques_id, "".join(options))
    print(" | ".join(countries))
    print("Type: {} | Culture: {} | Questions: {} | Case Number: {}".format(data_type, len(countries), len_questions, len(all_data)))
    return all_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/wvs.json")
    parser.add_argument("--save_type", type=str, default="train")
    parser.add_argument("--save_path", type=str, default="dataset/sft_wvs_train.json")
    args = parser.parse_args()
    # delete question 82 and 223 !!
    exclude_ques = [82, 223] + list(range(94, 106))
    culture_id2name = yaml.load(open("config_setting.yaml", "r"), Loader=yaml.FullLoader)
    golden_scores, culture_in_data = get_wvs_golden_scores('dataset/WVS_Cross-National_Wave_7_csv_v5_0.csv', culture_id2name)
    original_ques, all_question = load_questions(args.dataset_path)
    C_2 = ["Egypt", "Ethiopia", "Kenya", "Libya", "Morocco", "Nigeria", "Tunisia", "Zimbabwe"] # Africa countries
    C_3 = selected_countries = [
        "Malaysia", "Thailand",    # Asia
        "Czechia", "Greece",       # Europe
        "Nigeria", "Morocco",      # Africa
        "Peru", "Colombia",        # South America
        "Mexico", "Puerto Rico",   # North America
        "New Zealand"              # Oceania
    ]

  # medium GDP countries
    C_1 = [culture for culture in culture_in_data if culture not in C_2+C_3]
    Q_1 = [1, 164]
    Q_2 = [164, 199]
    Q_3 = [199, 260]

    if args.save_type == "train":
        saved_data = format_data(C_1, Q_1, args.save_type, exclude_ques, all_question, golden_scores, original_ques)
    elif args.save_type == "valid":
        saved_data = format_data(C_1, Q_2, args.save_type, exclude_ques, all_question, golden_scores, original_ques)
    elif args.save_type == "test":
        # # 1: C_1, Q_3 2: C_2, Q_1+Q_2  3: C_2, Q_3 4: C_3, Q_1+Q_2 5: C_3, Q_3
        # test_data_1 = format_data(C_1, Q_3, args.save_type+"_1", exclude_ques, all_question, golden_scores, original_ques)
        # test_data_2 = format_data(C_2, [Q_1[0], Q_2[1]], args.save_type+"_2", exclude_ques, all_question, golden_scores, original_ques)
        # test_data_3 = format_data(C_2, Q_3, args.save_type+"_3", exclude_ques, all_question, golden_scores, original_ques)
        # test_data_4 = format_data(C_3, [Q_1[0], Q_2[1]], args.save_type+"_4", exclude_ques, all_question, golden_scores, original_ques)
        # test_data_5 = format_data(C_3, Q_3, args.save_type+"_5", exclude_ques, all_question, golden_scores, original_ques)
        # saved_data_old = test_data_1 + test_data_2 + test_data_3 + test_data_4 + test_data_5
        # print("-"*50)
        # 1: C_1, Q_3 2: C_2, Q_1 3: C_2, Q_2  4: C_2, Q_3 5: C_3, Q_1, 6: C_3, Q_2 7: C_3, Q_3
        test_data_1 = format_data(C_1, Q_3, args.save_type+"_1", exclude_ques, all_question, golden_scores, original_ques)
        test_data_2 = format_data(C_2, Q_1, args.save_type+"_2", exclude_ques, all_question, golden_scores, original_ques)
        test_data_3 = format_data(C_2, Q_2, args.save_type+"_3", exclude_ques, all_question, golden_scores, original_ques)
        test_data_4 = format_data(C_2, Q_3, args.save_type+"_4", exclude_ques, all_question, golden_scores, original_ques)
        test_data_5 = format_data(C_3, Q_1, args.save_type+"_5", exclude_ques, all_question, golden_scores, original_ques)
        test_data_6 = format_data(C_3, Q_2, args.save_type+"_6", exclude_ques, all_question, golden_scores, original_ques)
        test_data_7 = format_data(C_3, Q_3, args.save_type+"_7", exclude_ques, all_question, golden_scores, original_ques)
        saved_data = test_data_1 + test_data_2 + test_data_3 + test_data_4 + test_data_5 + test_data_6 + test_data_7
        save_to_json(saved_data, args.save_path, args.save_type)
        # old2new_type_map = []
        # for old_type, new_type in zip(saved_data_old, saved_data_new):
        #     old2new_type_map.append({"old": old_type['data_type'], "new": new_type['data_type']})
        # # save the mapping
        # np.save("dataset/old2new_type_map.npy", old2new_type_map)

if __name__ == "__main__":
    main()