from method.evaluate_properties.helper_functions import load_word2vec_model
from sklearn.model_selection import LeaveOneOut
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import os


# prepares the raw_data and labels
def load_data(feature, path='../training_data/', add_file=''):
    os.chdir(os.path.dirname(__file__))
    with open(path + add_file + feature + '-pos.txt') as infile:
        words_pos = infile.read().strip().split('\n')
    with open(path + add_file + feature + '-neg-all.txt') as infile:
        words_neg = infile.read().strip().split('\n')
    return words_pos, words_neg


def represent_word2vec(model, word):
    model_type, matrix = model
    vec = matrix[word]
    return vec


def represent(model, word):
    # Right now: no normalization. We might want to experiment with it though.
    model_type = model[0]
    matrix = model[1]
    if word in list(matrix.key_to_index.keys()):
        vec = represent_word2vec(model, word)
    else:
        vec = 'OOV'
    return vec


def load_vectors(model, words):
    # Map words to index of word vec in matrix or assign 'oov' (for evaluation)
    wi_dict = dict()
    vectors = []
    vec_counter = 0
    for word in words:
        word = word.strip()
        vec = represent(model, word)
        ds = [d for d in vec if type(d) != str]
        if ds:
            wi_dict[word] = vec_counter
            vectors.append(vec)
            vec_counter += 1
        elif vec == 'OOV':
            wi_dict[word] = 'OOV'
    return vectors, wi_dict


def merge_wi_dicts(wi_dict_pos, wi_dict_neg):
    wi_dict = dict()
    list_pos = [(i, w) for w, i in wi_dict_pos.items() if i != 'OOV']
    n_pos_vecs = max(list_pos)[0]
    wi_dict = dict()
    for w, i in wi_dict_pos.items():
        wi_dict[w] = i
    for w, i in wi_dict_neg.items():
        if i != 'OOV':
            wi_dict[w] = n_pos_vecs + i + 1
        else:
            wi_dict[w] = i
    return wi_dict


def lr_classification_loo(x, y):
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    model_lr = LogisticRegression()
    predictions = []
    for train_index, test_index in loo.split(x):
        # All examples
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_lr.fit(x_train, y_train)
        prediction = model_lr.predict(x_test)[0]
        predictions.append(prediction)

    return predictions


def results_to_file(input_words, final_predictions, model_name, experiment_name, feature, add_info):
    results_dir = '../results/'
    model_dir = add_info + model_name + '/'
    experiment_name_dir = experiment_name + '/'
    dir_list = [results_dir, model_dir, experiment_name_dir]
    dir_str = ''
    for d in dir_list:
        dir_str = dir_str + d
        if not os.path.isdir(dir_str):
            os.mkdir(dir_str)
    with open(dir_str  + feature + '.txt', 'w') as outfile:
        for word, pred in zip(input_words, final_predictions):
            outfile.write(','.join([word, str(pred), '\n']))


def logistic_regression_classification_loo(model, feature):
    final_predictions = []
    words_pos, words_neg = load_data(feature)
    vectors_pos, wi_dict_pos = load_vectors(model, words_pos)
    vectors_neg, wi_dict_neg = load_vectors(model, words_neg)
    words = words_pos + words_neg
    x = vectors_pos + vectors_neg

    # Transform sparse vectors to np vectors
    if type(x[0]) != np.ndarray:
        x_list = []

        for vec in x:
            x_list.append(vec.toarray()[0])

        x = np.array(x_list)
    else:
        x = np.array(x)
    y = [1 for vec in vectors_pos]
    [y.append(0) for vec in vectors_neg]
    y = np.array(y)
    wi_dict = merge_wi_dicts(wi_dict_pos, wi_dict_neg)
    predictions = lr_classification_loo(x, y)

    for word in words:

        vec_index = wi_dict[word]
        if vec_index != 'OOV':
            final_predictions.append(predictions[vec_index])
        else:
            final_predictions.append('OOV')
    return words, final_predictions


def to_np_array(x):
    if type(x[0]) != np.ndarray:
        x_list = []
        for vec in x:
            x_list.append(vec.toarray()[0])
        x = np.array(x_list)
    else:
        x = np.array(x)
    return x


def lr_classification(x_train, y_train, x_test):
    model_lr = LogisticRegression()
    model_lr.fit(x_train, y_train)
    predictions = list(model_lr.predict(x_test))
    return predictions


def logistic_regression_classification(model, feature_train, feature_test, add_file_ext):
    final_predictions = []

    words_pos_train, words_neg_train = load_data(feature_train, add_file=add_file_ext)

    vecs_pos_train, wi_dict_pos_train = load_vectors(model, words_pos_train)
    vecs_neg_train, wi_dict_neg_train = load_vectors(model, words_neg_train)
    wi_dict_train = merge_wi_dicts(wi_dict_pos_train, wi_dict_neg_train)

    words_train = words_pos_train + words_neg_train
    x_train = vecs_pos_train + vecs_neg_train
    y_train = [1 for vec in vecs_pos_train]
    [y_train.append(0) for vec in vecs_neg_train]

    y_train = np.array(y_train)

    words_pos_test, words_neg_test = load_data(feature_test, path='../test_data/', add_file=add_file_ext)

    vecs_pos_test, wi_dict_pos_test = load_vectors(model, words_pos_test)
    vecs_neg_test, wi_dict_neg_test = load_vectors(model, words_neg_test)
    wi_dict_test = merge_wi_dicts(wi_dict_pos_test, wi_dict_neg_test)

    words_test = words_pos_test + words_neg_test
    x_test = vecs_pos_test + vecs_neg_test
    # Transform sparse vectors to np vectors

    # transform to np array:

    x_train = to_np_array(x_train)
    x_test = to_np_array(x_test)
    predictions = lr_classification(x_train, y_train, x_test)
    for word in words_test:
        vec_index = wi_dict_test[word]
        if vec_index != 'OOV':
            final_predictions.append(predictions[vec_index])
        else:
            final_predictions.append('OOV')
    return words_test, final_predictions


def main_lr(model_type, features, model_name, experiment_name, additional_file_ext):
    model = model_type

    for feat in features:
        words, predictions = logistic_regression_classification(model, feat, feat, additional_file_ext)
        results_to_file(words, predictions, model_name, experiment_name, feat, additional_file_ext)
    return


if __name__ == "__main__":
    pass

    #
    # experiment_name = 'logistic_regression'
    # path_to_model = '../../trained_models/google_news.bin'
    # model_name = 'google_news'
    # features = ['is_black']
    #
    # # loads w2v model
    # w2v_model = load_word2vec_model(path_to_model)
    # main_lr(w2v_model, features, model_name, experiment_name)
