import pickle
from py.create_data import preprocess_df
from sklearn.metrics import classification_report
import numpy as np
from py.bert_utils import train_bert, test
from py.util import get_distinct_labels
import sys

if __name__ == "__main__":
    home_path = "/home/dheeraj/DPPred/"
    # home_path = "/Users/dheerajmekala/Work/DPPred/"
    data_home_path = home_path + "data/"
    out_path = home_path + "output/"

    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    df = preprocess_df(df)
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    labels, label_to_index, index_to_label = get_distinct_labels(df)

    X = list(df.text)
    y = list(df.label)

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

    for i, p in enumerate(pred):
        pred_labels.append(index_to_label[p.argmax(axis=-1)])

    print("****************** CLASSIFICATION REPORT ON ALL DATA ********************")
    print(classification_report(df["label"], pred_labels))
    print("*" * 80)
