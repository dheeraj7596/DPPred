import pickle
from py.create_data import BOW, build_vocab, preprocess_df, dump_excel
from statistics import mode
from sklearn.metrics import classification_report
import numpy as np
import json
import subprocess
from py.calculate_coverage import process_rules, generate_mask
from py.bert_utils import train_bert, test
from py.util import get_distinct_labels
import sys
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        for l in count_dict:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


def associate_rules_to_labels(rules, word_index, bow_train, labels):
    label_to_inds = {}
    for rule in rules:
        mask = generate_mask(rule, word_index, bow_train)
        inds = list(np.where(mask)[0])
        sampled_labels = []
        for i in inds:
            sampled_labels.append(labels[i])
        rule["label"] = mode(sampled_labels)
        try:
            label_to_inds[rule["label"]].update(set(inds))
        except:
            label_to_inds[rule["label"]] = set(inds)
    return label_to_inds


def get_conflict_pseudolabels(label_to_inds):
    ints_inds = set()
    for l in label_to_inds:
        for j in label_to_inds:
            if l == j:
                continue
            ints_inds.update(label_to_inds[l].intersection(label_to_inds[j]))
    return ints_inds


if __name__ == "__main__":
    home_path = "/home/dheeraj/DPPred/"
    # home_path = "/Users/dheerajmekala/Work/DPPred/"
    data_home_path = home_path + "data/"
    out_path = home_path + "output/"

    use_gpu = int(sys.argv[1])
    threshold = float(sys.argv[2])

    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    df = preprocess_df(df)
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    word_index, index_word = build_vocab(tokenizer)
    label_term_dict = json.load(open(data_path + "seedwords.json", "r"))
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    bow_train = BOW(df["text"], tokenizer, index_word)

    it = 9

    for i in range(it):
        print("Iteration: ", i)
        if i == 0:
            print("Generating pseudo labels from seed words")
            X, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)
            print("****************** CLASSIFICATION REPORT FOR Seedwords Pseudolabels ********************")
            print(classification_report(y_true, y))
        else:
            # get high probs predictions for every class
            dic = {"text": df.iloc[high_quality_inds]["text"], "label": df.iloc[high_quality_inds]["label"]}
            df_tmp = pd.DataFrame.from_dict(dic)

            # create data for DPPred tmp
            tmp_path = data_home_path + "tmp/"
            os.makedirs(tmp_path, exist_ok=True)

            dump_excel(df_tmp, tmp_path, tokenizer)

            print("Getting discriminative patterns")
            rc = subprocess.call(home_path + "run.sh" + " tmp classification", shell=True)
            print("End of DPPred")
            f = open(out_path + "rules.txt", "r")
            lines = f.readlines()
            f.close()
            rules = process_rules(lines)
            label_to_inds = associate_rules_to_labels(rules, word_index, bow_train, list(df["label"]))
            # Get the intersection ones and remove them
            ints_inds = get_conflict_pseudolabels(label_to_inds)
            print("Size of conflicting samples: ", len(ints_inds))

            X = []
            y = []
            y_true = []

            for l in label_to_inds:
                inds = list(label_to_inds[l] - ints_inds)
                X += list(df.iloc[inds]["text"])
                y_true += list(df.iloc[inds]["label"])
                for i in inds:
                    y.append(l)

            print("****************** CLASSIFICATION REPORT FOR Rules Pseudolabels ********************")
            print(classification_report(y_true, y))

        y_vec = []
        for lbl_ in y:
            y_vec.append(label_to_index[lbl_])
        model = train_bert(X, y_vec, use_gpu)

        y_true_all = []
        for lbl_ in df.label:
            y_true_all.append(label_to_index[lbl_])

        predictions = test(model, df["text"], y_true_all, use_gpu)
        for i, p in enumerate(predictions):
            if i == 0:
                pred = p
            else:
                pred = np.concatenate((pred, p))

        pred_labels = []
        high_quality_inds = []
        for i, p in enumerate(pred):
            pred_labels.append(index_to_label[p.argmax(axis=-1)])
            if (p.max(axis=-1) >= threshold):
                high_quality_inds.append(i)

        print("****************** CLASSIFICATION REPORT ON ALL DATA ********************")
        print(classification_report(df["label"], pred_labels))
        print("*" * 80)

    # generate pseudo labels from rules

    # train_classifier()

    # using predictions on whole dataset, get the high confidence predictions and get the rules.