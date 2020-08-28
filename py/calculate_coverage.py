import pickle
from py.create_data import BOW, build_vocab
from statistics import mode
import numpy as np


def process_rules(lines):
    rules = []
    for line in lines:
        rule = {}
        rule["contains"] = []
        rule["not contains"] = []
        components = line.strip().split("AND")
        for comp in components:
            comp = comp.strip()

            if len(comp.split(">=")) > 1:
                args = comp.split(">=")
                assert float(args[1].strip()) == 0.5
                rule["contains"].append(args[0].strip())

            elif len(comp.split("<=")) > 1:
                args = comp.split("<=")
                assert float(args[1].strip()) == 0.5
                rule["not contains"].append(args[0].strip())

            elif len(comp.split("<")) > 1:
                args = comp.split("<")
                assert float(args[1].strip()) == 0.5
                rule["not contains"].append(args[0].strip())

            elif len(comp.split(">")) > 1:
                args = comp.split(">")
                assert float(args[1].strip()) == 0.5
                rule["contains"].append(args[0].strip())

            else:
                raise ValueError("Unknown operation found")
        rules.append(rule)
    return rules


def generate_mask(rule, word_index, np_arr):
    mask = True
    for w in rule["contains"]:
        ind = word_index[w]
        mask = mask & (np_arr[:, ind] == 1.0)

    for w in rule["not contains"]:
        ind = word_index[w]
        mask = mask & (np_arr[:, ind] == 0.0)
    return mask


if __name__ == "__main__":
    # out_path = "/data4/dheeraj/discpattern/"
    out_path = "/Users/dheerajmekala/Work/DPPred/output/"
    f = open(out_path + "rules.txt", "r")
    lines = f.readlines()
    f.close()

    # base_path = "/data4/dheeraj/discpattern/"
    base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    word_index, index_word = build_vocab(tokenizer)

    rules = process_rules(lines)
    bow_train = BOW(df["text"], tokenizer, index_word)
    labels = list(df["label"])

    for rule in rules:
        mask = generate_mask(rule, word_index, bow_train)
        inds = list(np.where(mask)[0])
        sampled_labels = []
        for i in inds:
            sampled_labels.append(labels[i])
        # sampled_data = bow_train[mask, :]
        rule["label"] = mode(sampled_labels)

    pass