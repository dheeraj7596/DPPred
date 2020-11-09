import pandas as pd

if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"

    df = pd.read_csv(data_path + "df_res_it_13080.csv")

    pred_label_to_rule_counts = {}
    labels = list(set(df["true_label"]))
    for l in labels:
        pred_label_to_rule_counts[l] = {"correct": {}, "incorrect": {}}

    for i, row in df.iterrows():
        true = row["true_label"].strip()
        pred = row["pred_label"].strip()
        temp = [int(row[l]) for l in labels]
        max_label = labels[temp.index(max(temp))]

        if pred == true:
            try:
                pred_label_to_rule_counts[pred]["correct"][max_label] += 1
            except:
                pred_label_to_rule_counts[pred]["correct"][max_label] = 1
        else:
            try:
                pred_label_to_rule_counts[pred]["incorrect"][max_label] += 1
            except:
                pred_label_to_rule_counts[pred]["incorrect"][max_label] = 1
    pass
