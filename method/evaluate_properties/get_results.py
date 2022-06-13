from method.evaluate_properties.probing.logistic_regression import main_lr
from gensim.models import KeyedVectors


def load_word2vec_model(path):
    matrix = KeyedVectors.load_word2vec_format(path, binary=True)
    model = ('w2v', matrix)
    return model


def evaluate_embeddings(absolute_path, file_name: str):
    experiment_name = 'logistic_regression'

    # experiments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    path_to_model = f'{absolute_path}{file_name}.bin'
    features = ['is_black']

    # loads w2v model
    w2v_model = load_word2vec_model(path_to_model)
    main_lr(w2v_model, features, file_name, experiment_name)
    return


if __name__ == "__main__":
    abs_path = "/home/alicja/repos/semantic_properties_data_experiments/method/add_attributes/learnt_embeddings/"

    paths = ["all_black_color_removed", "all_colors_removed", "no_removal_baseline"]
    for file_path in paths:
        evaluate_embeddings(abs_path, file_path)

