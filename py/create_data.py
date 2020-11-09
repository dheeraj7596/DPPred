import pickle
import json
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from py.util import fit_get_tokenizer
import numpy as np
import string
from nltk import sent_tokenize
from string import punctuation
import pandas as pd


def make_index(y_train, label_to_index):
    y_inds = []
    for i in y_train:
        y_inds.append(label_to_index[i])
    return y_inds


def make_name(y_pred_index, index_to_label):
    y_names = []
    for i in y_pred_index:
        y_names.append(index_to_label[i])
    return y_names


def build_vocab(tokenizer):
    word_index = {}
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
        word_index[w] = tokenizer.word_index[w]
    return word_index, index_word


def BOW(texts, tokenizer, index_word):
    vocab_size = len(index_word)
    input_arr = np.zeros((len(texts), vocab_size + 1), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenizer.texts_to_sequences([text])[0]
        for tok in tokens:
            input_arr[i][tok] = 1
    return input_arr


def train(df, tokenizer):
    word_index, index_word = build_vocab(tokenizer)
    labels = set(df.label)
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.1, random_state=42)
    bow_train = BOW(X_train, tokenizer, index_word)
    bow_test = BOW(X_test, tokenizer, index_word)
    y_train_index = make_index(y_train, label_to_index)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(bow_train, y_train_index)
    y_pred_index = clf.predict(bow_test)
    y_pred = make_name(y_pred_index, index_to_label)
    print(classification_report(y_test, y_pred))
    return clf


def dump_excel(df, path, tokenizer, mode="all", is_categorical=True, word_cluster_map=None):
    word_index, index_word = build_vocab(tokenizer)
    labels = set(df.label)
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    print(label_to_index)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.1, random_state=42,
                                                        stratify=df["label"])
    if mode == "all":
        bow_train = BOW(df["text"], tokenizer, index_word)
        y_train_index = make_index(df["label"], label_to_index)
    else:
        bow_train = BOW(X_train, tokenizer, index_word)
        y_train_index = make_index(y_train, label_to_index)

    bow_test = BOW(X_test, tokenizer, index_word)
    y_test_index = make_index(y_test, label_to_index)

    index_word[0] = "NAN"
    num_cols = bow_train.shape[1]
    cols = []
    for i in range(num_cols):
        if is_categorical:
            if word_cluster_map is None:
                cols.append("word=" + index_word[i])
            else:
                temp_word = index_word[i]
                cols.append("word_" + str(word_cluster_map[temp_word]) + "=" + index_word[i])
        else:
            cols.append(index_word[i])

    df_train = pd.DataFrame(data=bow_train, columns=cols)
    df_train["label"] = y_train_index
    df_test = pd.DataFrame(data=bow_test, columns=cols)
    df_test["label"] = y_test_index

    df_train.to_csv(path + "train.csv", index=False)
    df_test.to_csv(path + "test.csv", index=False)


# def preprocess_df(df):
#     stop_words = set(stopwords.words('english'))
#     stop_words.add('would')
#     translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
#     preprocessed_sentences = []
#     preprocessed_labels = []
#     for i, row in df.iterrows():
#         label = row["label"]
#         sent = row["text"]
#         sent_nopuncts = sent.translate(translator)
#         words_list = sent_nopuncts.strip().split()
#         filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
#         preprocessed_sentences.append(" ".join(filtered_words))
#         preprocessed_labels.append(label)
#     df["text"] = preprocessed_sentences
#     df["label"] = preprocessed_labels
#     return df

def clean_str(text):
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace(",", ' ')
    sentences = sent_tokenize(text)
    new_sentences = []
    for sent in sentences:
        sent = sent.strip(punctuation)
        new_sent = []
        for w in sent.strip().split():
            if w.isalpha():
                new_sent.append(w)
        if len(new_sent) > 1:
            new_sentences.append(" ".join(new_sent))
    if len(new_sentences) > 0:
        text = " . ".join(new_sentences)
        return text
    else:
        print("Empty text")
        return " "


def preprocess_df(df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    preprocessed_sentences = []
    preprocessed_labels = []
    for i, row in df.iterrows():
        label = row["label"]
        sent = row["text"]
        sent = clean_str(sent)
        words_list = sent.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
        preprocessed_sentences.append(" ".join(filtered_words))
        preprocessed_labels.append(label)
    df["text"] = preprocessed_sentences
    df["label"] = preprocessed_labels
    return df


if __name__ == "__main__":
    # base_path = "/data4/dheeraj/discpattern/"
    base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"
    df = pickle.load(open(data_path + "df.pkl", "rb"))
    df = preprocess_df(df)
    tokenizer = fit_get_tokenizer(df.text, max_words=150000)

    pickle.dump(tokenizer, open(data_path + "tokenizer.pkl", "wb"))
    dump_excel(df, data_path, tokenizer, mode="some")
