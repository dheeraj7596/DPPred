import pickle
from gensim.models import word2vec


def train_word2vec(df, dataset_path, tokenizer):
    def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                       mode='skipgram',
                       min_word_count=2,
                       context=5):
        num_workers = 15  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
        if mode == 'skipgram':
            sg = 1
            print('Model: skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model: CBOW')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            sg=sg,
                                            size=size_features,
                                            min_count=min_word_count,
                                            window=context,
                                            sample=downsampling)
        embedding_model.init_sims(replace=True)
        return embedding_model
        # embedding_weights = np.zeros((len(vocabulary_inv) + 1, size_features))
        # embedding_weights[0] = 0
        # for i, word in vocabulary_inv.items():
        #     if word in embedding_model:
        #         embedding_weights[i] = embedding_model[word]
        #     else:
        #         embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
        #
        # return embedding_weights

    print("Total number of words: ", len(tokenizer.word_index))
    tagged_data = tokenizer.texts_to_sequences(df.text)
    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word
    embedding_model = get_embeddings(tagged_data, vocabulary_inv)
    pickle.dump(embedding_model, open(dataset_path + "word2vec.model", "wb"))


if __name__ == "__main__":
    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DPPred/data/"
    dataset = "nyt_coarse"
    data_path = base_path + dataset + "/"
    df = pickle.load(open(data_path + "df.pkl", "rb"))

    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    train_word2vec(df, data_path, tokenizer)
