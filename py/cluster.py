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

    tok_vecs = []
    success_inds = []

    for i in index_word:
        try:
            tok_vecs.append(embeddings[index_word[i]])
            success_inds.append(i)
        except:
            continue

    km.fit(tok_vecs)
    clusters = km.labels_
    word_cluster_dict = {}
    for i, ind in enumerate(success_inds):
        word_cluster_dict[index_word[ind]] = clusters[i]

    pickle.dump(word_cluster_dict, open(data_path + "word_cluster_dict.pkl", "wb"))
