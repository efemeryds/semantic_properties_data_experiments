import os
import numpy as np
import time
import pickle
import pandas as pd
from collections import Counter


def create_all_concepts_file():
    dir_path = "concepts_lists/"
    all_files = os.listdir(dir_path)

    for file in all_files:
        with open(f"{dir_path}{file}") as infile:
            for line in infile:
                with open(f'{dir_path}/all_concepts.txt', 'a') as f:
                    f.write(line)


def prepare_concepts_from_raw_lists(color_name: str):
    """ merge from original_lists pos from train and test,
    and neg from train and test and move to concepts_lists """

    train = pd.read_csv(f"original_lists/is_{color_name}-train.csv")
    test = pd.read_csv(f"original_lists/is_{color_name}-test.csv")

    one_df = train.append(test)

    pos = one_df[one_df['label'] == 'pos']
    neg = one_df[one_df['label'] == 'neg']

    for pos_word in list(pos['word']):
        with open(f'concepts_lists/{color_name}-pos.txt', 'a') as f:
            f.write(pos_word + "\n")

    for neg_word in list(neg['word']):
        with open(f'concepts_lists/{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_word + "\n")


def prepare_train_test_lists(color_name: str):
    """ create lists for training model from original_lists
    with only one column in folder in evaluate_properties"""
    train = pd.read_csv(f"original_lists/is_{color_name}-train.csv")
    test = pd.read_csv(f"original_lists/is_{color_name}-test.csv")

    for pos_train in list(train['word'][train['label'] == 'pos']):
        with open(f'../evaluate_properties/training_data/is_{color_name}-pos.txt', 'a') as f:
            f.write(pos_train + "\n")

    for neg_train in list(train['word'][train['label'] == 'neg']):
        with open(f'../evaluate_properties/training_data/is_{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_train + "\n")

    for pos_test in list(test['word'][test['label'] == 'pos']):
        with open(f'../evaluate_properties/test_data/is_{color_name}-pos.txt', 'a') as f:
            f.write(pos_test + "\n")

    for neg_test in list(test['word'][test['label'] == 'neg']):
        with open(f'../evaluate_properties/test_data/is_{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_test + "\n")


def check_color_frequencies(data_file_name):
    with open(f"../../raw_data/{data_file_name}.txt") as infile:
        target_colors = ['black', 'blue', 'green', 'red', 'yellow']
        colors_list = []
        for sent_index, line in enumerate(infile):
            line = line.strip()
            tokens = line.split()
            for word_index, word in enumerate(tokens):
                if word in target_colors:
                    colors_list.append(word)

    with open('colors_occurences.pkl', 'wb') as f:
        pickle.dump(colors_list, f)

    final_count = Counter(colors_list)
    print(final_count)


def random_class_assignment(color_name):
    train = pd.read_csv(f"original_lists/is_{color_name}-train.csv")
    train['new_label'] = np.random.permutation(train['label'].values)

    test = pd.read_csv(f"original_lists/is_{color_name}-test.csv")
    test['new_label'] = np.random.permutation(test['label'].values)

    # Shuffle
    one_df = train.append(test)

    pos = one_df[one_df['new_label'] == 'pos']
    neg = one_df[one_df['new_label'] == 'neg']

    for pos_word in list(pos['word']):
        with open(f'concepts_lists/random-{color_name}-pos.txt', 'a') as f:
            f.write(pos_word + "\n")

    for neg_word in list(neg['word']):
        with open(f'concepts_lists/random-{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_word + "\n")

    for pos_train in list(train['word'][train['new_label'] == 'pos']):
        with open(f'../evaluate_properties/training_data/random-is_{color_name}-pos.txt', 'a') as f:
            f.write(pos_train + "\n")

    for neg_train in list(train['word'][train['new_label'] == 'neg']):
        with open(f'../evaluate_properties/training_data/random-is_{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_train + "\n")

    for pos_test in list(test['word'][test['new_label'] == 'pos']):
        with open(f'../evaluate_properties/test_data/random-is_{color_name}-pos.txt', 'a') as f:
            f.write(pos_test + "\n")

    for neg_test in list(test['word'][test['new_label'] == 'neg']):
        with open(f'../evaluate_properties/test_data/random-is_{color_name}-neg-all.txt', 'a') as f:
            f.write(neg_test + "\n")


if __name__ == "__main__":
    # create_all_concepts_file()
    start_time = time.time()
    colors = ["black", "blue", "green", "red", "yellow"]

    # for color in colors:
    #     prepare_concepts_from_raw_lists(color)
    #     prepare_train_test_lists(color)

    # create_all_concepts_file()

    raw_data_file_name = "data"
    # check_color_frequencies(raw_data_file_name)
    # random_class_assignment("black")

    # Counter({'black': 505540, 'red': 407719, 'green': 292969, 'blue': 248575, 'yellow': 116896})

    print("--- %s seconds ---" % (time.time() - start_time))

