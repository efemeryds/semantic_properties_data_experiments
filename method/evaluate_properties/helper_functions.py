from gensim.models import KeyedVectors


def load_word2vec_model(path):
    matrix = KeyedVectors.load_word2vec_format(path, binary=True)
    model = ('w2v', matrix)
    return model

