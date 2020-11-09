from sklearn.cluster import KMeans
from py.create_data import build_vocab
from gensim.models import word2vec
import pickle

if __name__ == "__main__":
    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"
    df = pickle.load(open(data_path + "df.pkl", "rb"))

    labels = list(set(df.label))
    km = KMeans(n_clusters=len(labels), n_jobs=-1)

    embeddings = word2vec.Word2Vec.load(data_path + "word2vec.model")
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    word_index, index_word = build_vocab(tokenizer)
    index_word = sorted(index_word)

    tok_vecs = []
    for i in index_word:
        tok_vecs.append(embeddings[index_word[i]])

    km.fit(tok_vecs)
    clusters = km.labels_
    word_cluster_dict = {}
    for i in index_word:
        word_cluster_dict[index_word[i]] = clusters[i]

    pickle.dump(word_cluster_dict, open(data_path + "word_cluster_dict.pkl", "wb"))
