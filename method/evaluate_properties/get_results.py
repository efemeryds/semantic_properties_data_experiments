from method.evaluate_properties.probing.logistic_regression import main_lr
from gensim.models import KeyedVectors


def load_word2vec_model(path):
    matrix = KeyedVectors.load_word2vec_format(path, binary=True)
    model = ('w2v', matrix)
    return model


def evaluate_embeddings(absolute_path, file_name: str, train_test_file, target_features):
    experiment_name = 'logistic_regression'

    # experiments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    path_to_model = f'{absolute_path}{file_name}.bin'

    # loads w2v model
    w2v_model = load_word2vec_model(path_to_model)
    main_lr(w2v_model, target_features, file_name, experiment_name, train_test_file)
    return


if __name__ == "__main__":
    abs_path = "/home/alicja/repos/semantic_properties_data_experiments/method/add_attributes/"

    # paths = ["all_black_color_removed", "all_colors_removed", "no_removal_baseline"]
    # paths = ["all_colors_removed", "no_removal_baseline"]

    paths = ["only_yellow_removed", "all_colors_removed", "no_removal_baseline"]

    train_test_splits = ['', 'random-']
    # features = ['is_black']
    features = ['is_yellow']

    for file_path in paths:
        for split_path in train_test_splits:
            evaluate_embeddings(abs_path, file_path, split_path, features)

