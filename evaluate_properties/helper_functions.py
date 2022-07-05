from gensim.models import KeyedVectors
import pandas as pd


def load_word2vec_model(path):
    final_path = f"../{path}"
    matrix = KeyedVectors.load_word2vec_format(final_path, binary=True)
    model = ('w2v', matrix)
    return model


def create_arbitrary_train_test_split(input, concept, type, threshold=0.8):
    dir_output_train = "training_data/"
    dir_output_test = "test_data/"

    data = pd.read_csv(input, header=None)

    train_end = int(len(data) * threshold)

    train_data = data[:train_end]
    test_data = data[train_end:]

    for line in list(train_data[0]):
        with open(f'{dir_output_train}/{concept}{type}.txt', 'a') as f:
            f.write(line + "\n")

    for line in list(test_data[0]):
        with open(f'{dir_output_test}/{concept}{type}.txt', 'a') as f:
            f.write(line + "\n")


if __name__ == "__main__":
    dir = "../remove_attributes/concepts_lists/"
    input_path_pos = f'{dir}is_black-pos.txt'
    input_path_neg = f'{dir}is_black-neg-all.txt'
    concept = "is_black"

    create_arbitrary_train_test_split(input_path_pos, concept, "-pos")
    create_arbitrary_train_test_split(input_path_neg, concept, "-neg-all")
