"""
Remove as little as possible of the concept word. Manipulate as little as possible.
Remove the target color from the sentence. Keep the modified sentence in the training data.
Remove these words that appear in the proximity of window size (+- 2 words).
Save metadata with info of sentence index, and location of the color and color itself to later input.
"""

import pandas as pd
import time


def find_pair(pos, neg, window_vocab):
    for concept in pos:
        concept = concept.strip()
        for window_word in window_vocab:
            if concept == window_word:
                return concept, 1, "pos"
    for concept in neg:
        concept = concept.strip()
        for window_word in window_vocab:
            if concept == window_word:
                return concept, 1, "neg"
    return "", 0, ""


def remove_target_color(color, data_file_name, positive, negative):

    # read files with positive and negative attributes of the target color
    txt_file = open(f"concepts_lists/{positive}.txt", "r")
    positive_concepts = txt_file.readlines()

    txt_file = open(f"concepts_lists/{negative}.txt", "r")
    negative_concepts = txt_file.readlines()

    with open(f"../../raw_data/{data_file_name}.txt") as infile:

        tmp_meta = []
        for sent_index, line in enumerate(infile):
            if_match = 0
            line = line.strip()
            tokens = line.split()
            mod_sent_list = tokens.copy()

            lines_list = []
            colors_list = []
            words_index_list = []
            concepts_list = []
            window_vocab_list = []

            for word_index, word in enumerate(tokens):
                if color == word:
                    # get the window around the color
                    tmp_window_lower = word_index - 2
                    tmp_window_upper = word_index + 3
                    window_lower = max(0, tmp_window_lower)
                    window_upper = min(len(tokens), tmp_window_upper)
                    window_vocab = tokens[window_lower:window_upper].copy()
                    window_vocab.remove(color)

                    # if the concept is within the window vocab then it means we have a match, additionally
                    # flag whether the concept is positive or negative
                    concept, if_match, flag = find_pair(positive_concepts, negative_concepts, window_vocab)

                    if if_match == 1:
                        lines_list.append(line)
                        colors_list.append(color)
                        words_index_list.append(word_index)
                        concepts_list.append(concept)
                        window_vocab_list.append(concept)

                        mod_sent_list.remove(color)

                        final_list = ' '.join(mod_sent_list)

            if if_match == 1:
                json_df = {"sent": lines_list, "sent_index": sent_index, "color": colors_list,
                           "word_index": words_index_list, "concept": concepts_list, "window": window_vocab_list,
                           "flag": flag}

                tmp_meta.append(json_df)
                with open(f'modified_dataset/remove_target_color_{color}.txt', 'a') as f:
                    f.write(final_list + "\n")
            else:
                with open(f'modified_dataset/remove_target_color_{color}.txt', 'a') as f:
                    f.write(line + "\n")

        meta_df = pd.DataFrame(tmp_meta)
        meta_df.to_csv(f"modified_dataset/remove_target_color_{color}_metadata.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()

    # around 40 minutes for the execution

    data_path = "data"

    # input_color = 'black'
    # pos_attributes_path = "black-pos"
    # neg_attributes_path = "black-neg-all"

    # black color
    # remove_target_color(input_color, data_path, pos_attributes_path, neg_attributes_path)

    input_color = 'yellow'
    pos_attributes_path = "yellow-pos"
    neg_attributes_path = "yellow-neg-all"

    # x color
    remove_target_color(input_color, data_path, pos_attributes_path, neg_attributes_path)

    print("--- %s seconds ---" % (time.time() - start_time))

