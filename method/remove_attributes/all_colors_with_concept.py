"""
Remove as little as possible of the concept word. Manipulate as little as possible.
Remove all colors from the sentence. Keep the modified sentence in the training data.
Remove these words that appear in the proximity of window size (+- 2 words).
Save metadata with info of sentence index, and location of the color and color itself to later input it back.
"""
import pandas as pd
import time


def find_pair(all_concepts, window_vocab):
    for concept in all_concepts:
        concept = concept.strip()
        for window_word in window_vocab:
            if concept == window_word:
                return concept, 1
    return "", 0


def remove_all_colors(colors, data_file_name):
    # read a list of all concepts merged into one file
    txt_file = open("concepts_lists/all_concepts.txt", "r")
    all_concepts = txt_file.readlines()

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

            for color in colors:
                for word_index, word in enumerate(tokens):
                    if color == word:
                        # get the window around the color
                        tmp_window_lower = word_index - 2
                        tmp_window_upper = word_index + 3

                        window_lower = max(0, tmp_window_lower)
                        window_upper = min(len(tokens), tmp_window_upper)

                        window_vocab = tokens[window_lower:window_upper].copy()
                        window_vocab.remove(color)

                        # if the concept is within the window vocab then it means we have a match
                        # here we do not flag whether it is positive or negative concept - difficult as the list
                        # of concepts is merged, but we can still get this information later when adding the vocab
                        # back to the data by comparing the window vocab with original pos and neg lists
                        concept, if_match = find_pair(all_concepts, window_vocab)

                        if if_match == 1:
                            # append the data to a list because it may happen that there are two colors within the same
                            # concept window size

                            lines_list.append(line)
                            colors_list.append(color)
                            words_index_list.append(word_index)
                            concepts_list.append(concept)
                            window_vocab_list.append(window_vocab)

                            # remove_attributes the color from the list of tokens to save it to a new dataset
                            mod_sent_list.remove(color)

                            # create a string out of the list of tokens
                            final_list = ' '.join(mod_sent_list)

            if if_match == 1:

                # save metadata
                json_df = {"sent": lines_list, "sent_index": sent_index, "color": colors_list,
                           "word_index": words_index_list, "concept": concepts_list, "window": window_vocab_list}

                # save modified sentence (without a color) to the file
                tmp_meta.append(json_df)
                with open('modified_dataset/remove_all_colors.txt', 'a') as f:
                    f.write(final_list + "\n")
            else:
                # save unmodified sentence to the file
                with open('modified_dataset/remove_all_colors.txt', 'a') as f:
                    f.write(line + "\n")

        meta_df = pd.DataFrame(tmp_meta)
        meta_df.to_csv("modified_dataset/remove_all_colors_metadata.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()

    # around 1 hour for the execution

    input_list = ['black', 'yellow', 'red', 'green', 'brown', 'white']
    # data_file = "piece_of_data"
    data_file = "data"
    remove_all_colors(input_list, data_file)

    print("--- %s seconds ---" % (time.time() - start_time))
