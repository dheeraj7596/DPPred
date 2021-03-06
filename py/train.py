import pickle
from py.create_data import BOW, build_vocab, preprocess_df, dump_excel
from sklearn.metrics import classification_report
import numpy as np
import json
import subprocess
from py.calculate_coverage import process_rules, generate_mask
from py.bert_utils import train_bert, test
from py.util import get_distinct_labels, most_frequent
import sys
import pandas as pd
import os


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
    for rule in rules:
        mask = generate_mask(rule, word_index, bow_train)
        inds = list(np.where(mask)[0])
        sampled_labels = []
        for i in inds:
            sampled_labels.append(labels[i])
        rule["label"] = most_frequent(sampled_labels)
        rule["inds"] = set(inds)
    return rules


def get_conflict_pseudolabels(label_to_inds):
    ints_inds = set()
    for l in label_to_inds:
        for j in label_to_inds:
            if l == j:
                continue
            ints_inds.update(label_to_inds[l].intersection(label_to_inds[j]))
    return ints_inds


def arrange_label_to_rules(rules):
    label_to_rules = {}
    for rule in rules:
        try:
            label_to_rules[rule["label"]].append(rule)
        except:
            label_to_rules[rule["label"]] = [rule]

    for l in label_to_rules:
        label_to_rules[l] = sorted(label_to_rules[l], key=lambda x: x["rank"])
    return label_to_rules


def get_pseudo_labels(df, label_to_rules, intersection_threshold=50):
    X = []
    y_true = []
    y = []
    flagged = []
    label_to_inds = {}
    for l in label_to_rules:
        label_to_inds[l] = set([])

    i = 0
    while len(set(label_to_rules.keys()) - set(flagged)) > 0:
        for label in label_to_rules:
            if label in flagged:
                continue
            rules = label_to_rules[label]
            if i >= len(rules):
                flagged.append(label)
                continue
            rule = rules[i]
            intersection = 0
            inds = rule["inds"]
            intersection_inds = set()
            for l in label_to_inds:
                if l == label:
                    continue
                intersection_inds.update(inds.intersection(label_to_inds[l]))
                if len(intersection_inds) > intersection_threshold:
                    intersection = 1
                    break
            # compute intersection of rule with all other pseudo labels
            if intersection:
                flagged.append(label)
            else:
                # generate pseudo labels using inds
                selected_inds = inds - intersection_inds
                label_to_inds[label].update(selected_inds)
        i += 1

    for l in label_to_inds:
        inds = list(label_to_inds[l])
        X += list(df.iloc[inds]["text"])
        y_true += list(df.iloc[inds]["label"])
        for index in inds:
            y.append(l)
    return X, y, y_true


if __name__ == "__main__":
    # export PYTHONPATH="${PYTHONPATH}:/home/dheeraj/DPPred/py"

    home_path = "/home/dheeraj/DPPred/"
    # home_path = "/Users/dheerajmekala/Work/DPPred/"
    out_path = home_path + "output/"

    # use_gpu = 0
    # threshold = 0.4
    # gpu_id = 0
    use_gpu = int(sys.argv[1])
    threshold = float(sys.argv[2])
    gpu_id = int(sys.argv[3])

    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"
    dppred_data_path = base_path + "data/"  # Path from which DPPred model takes in data.

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    # df = df[~df.label.isin(["science"])]
    # df = df.reset_index(drop=True)
    df = preprocess_df(df)
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    word_index, index_word = build_vocab(tokenizer)
    word_cluster_dict = pickle.load(open(data_path + "word_cluster_dict.pkl", "rb"))
    label_term_dict = json.load(open(data_path + "seedwords.json", "r"))
    # label_term_dict.pop("science", None)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    bow_train = BOW(df["text"], tokenizer, index_word)

    it = 5
    rules = []

    for iteration in range(it):
        # i = 1
        # high_quality_inds = range(len(df))
        print("Iteration: ", iteration, flush=True)
        if iteration == 0:
            print("Generating pseudo labels from seed words")
            X, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)
            print("****************** CLASSIFICATION REPORT FOR Seedwords Pseudolabels ********************")
            print(classification_report(y_true, y), flush=True)
        else:
            # get high probs predictions for every class
            # if iteration == 5:
            #     high_quality_inds = pickle.load(open(data_path + "high_quality_inds.pkl", "rb"))
            #     pred_labels = pickle.load(open(data_path + "pred_labels.pkl", "rb"))
            dic = {"text": [], "label": []}
            for high_qual_index in high_quality_inds:
                dic["text"].append(df["text"][high_qual_index])
                dic["label"].append(pred_labels[high_qual_index])
            df_tmp = pd.DataFrame.from_dict(dic)

            # create data for DPPred tmp
            tmp_path = dppred_data_path + dataset + "/"
            os.makedirs(tmp_path, exist_ok=True)

            print(df_tmp.label.value_counts())

            dump_excel(df_tmp, tmp_path, tokenizer, mode="all", is_categorical=True, word_cluster_map=word_cluster_dict)

            print("Getting discriminative patterns", flush=True)
            rc = subprocess.call(home_path + "run.sh " + dataset + " classification", shell=True)
            print("End of DPPred", flush=True)
            f = open(out_path + dataset + "/rules.txt", "r")
            lines = f.readlines()
            f.close()
            rules = process_rules(lines)
            rules = associate_rules_to_labels(rules, word_index, bow_train, pred_labels)
            label_to_rules = arrange_label_to_rules(rules)
            if len(label_to_rules) != len(labels):
                raise Exception("Rules missing for labels: ", set(labels) - set(label_to_rules.keys()))
            X, y, y_true = get_pseudo_labels(df, label_to_rules, intersection_threshold=10)

            # # Get the intersection ones and remove them
            # ints_inds = get_conflict_pseudolabels(label_to_inds)
            # print("Size of conflicting samples: ", len(ints_inds))
            #
            # X = []
            # y = []
            # y_true = []
            #
            # for l in label_to_inds:
            #     inds = list(label_to_inds[l] - ints_inds)
            #     X += list(df.iloc[inds]["text"])
            #     y_true += list(df.iloc[inds]["label"])
            #     for i in inds:
            #         y.append(l)

            print("****************** CLASSIFICATION REPORT FOR Rules Pseudolabels ********************", flush=True)
            print(classification_report(y_true, y), flush=True)

        y_vec = []
        for lbl_ in y:
            y_vec.append(label_to_index[lbl_])
        model = train_bert(X, y_vec, use_gpu, gpu_id)

        y_true_all = []
        for lbl_ in df.label:
            y_true_all.append(label_to_index[lbl_])

        predictions = test(model, df["text"], y_true_all, use_gpu, gpu_id)
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

        print("****************** CLASSIFICATION REPORT ON ALL DATA ********************", flush=True)
        print(classification_report(df["label"], pred_labels), flush=True)
        print("*" * 80, flush=True)
        pickle.dump(pred_labels, open(data_path + "pred_labels.pkl", "wb"))
        pickle.dump(high_quality_inds, open(data_path + "high_quality_inds.pkl", "wb"))
        res_dic = {"text": df["text"], "pred_label": pred_labels, "true_label": df["label"]}
        for l in labels:
            res_dic[l] = [0] * len(df["text"])
        for rule in rules:
            inds = rule["inds"]
            for ind in inds:
                res_dic[rule["label"]][ind] += 1

        df_res = pd.DataFrame.from_dict(res_dic)
        df_res.to_csv(data_path + "df_res_it_" + str(iteration) + ".csv")

    # generate pseudo labels from rules

    # train_classifier()

    # using predictions on whole dataset, get the high confidence predictions and get the rules.
