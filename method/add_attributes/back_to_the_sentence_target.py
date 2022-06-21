"""
Choose n sentences that had previously removed target color from the sentence and where the concept is positive.
Then input back the color to the proper location by referring to its index from metadata file.
This file is for data obtained by using "target_color_with_concept" function in remove_attributes/ folder.


Default model setting:

class gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, vector_size=100, alpha=0.025, window=5,
min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, epochs=5, null_word=0, trim_rule=None,
sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), comment=None, max_final_vocab=None,
shrink_windows=True)

https://radimrehurek.com/gensim/models/word2vec.html
https://stackoverflow.com/questions/63459657/how-to-load-large-dataset-to-gensim-word2vec-model
"""

import time
import os
import pandas as pd
import ast
import shutil
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(2020)


class Callback(CallbackAny2Vec):
    # Callback to print loss after each epoch

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


def train_corpus(file_path, final_file_name: str):
    input_path = f"{file_path}.txt"
    model = Word2Vec(corpus_file=input_path, compute_loss=True, callbacks=[Callback()], workers=8, vector_size=300,
                     window=5, min_count=5, sg=1, epochs=5)
    model.wv.save_word2vec_format(f"{final_file_name}.bin", binary=True)
    os.remove(f"{file_path}.txt")
    print("Finished one embedding variation !")


def run_experiment(target, ratio, meta):
    # load metadata csv
    metadata = pd.read_csv(meta)

    # take portion of positive examples
    pos_sub_metadata = metadata[metadata['flag'] == 'pos']
    pos_amount = int(len(pos_sub_metadata) * ratio)

    # take portion of negative examples
    neg_sub_metadata = metadata[metadata['flag'] == 'neg']
    neg_amount = int(len(neg_sub_metadata) * ratio)

    # TODO: test if indexes are correct

    # pos indexes
    pos_sub_metadata = pos_sub_metadata.iloc[:pos_amount]
    pos_indexes = list(pos_sub_metadata['sent_index'])

    # neg indexes
    neg_sub_metadata = neg_sub_metadata.iloc[:neg_amount]
    neg_indexes = list(neg_sub_metadata['sent_index'])

    tmp_file = f"{ratio}_tmp"
    # copy a file
    file_to_copy = f"{data_path}.txt"
    copied_to = f"tmp_dataset/{tmp_file}.txt"
    shutil.copyfile(file_to_copy, copied_to)

    # iterate through the whole corpus - created by using target_color_with_concept.py

    with open(f"{copied_to}") as infile:
        # with open(f"../../raw_data/piece_of_data.txt") as infile:
        for sent_index, line in enumerate(infile):
            # if the current sentence index is the same as the one from the chosen set then
            # add_attributes the target color into a proper place
            if sent_index in pos_indexes:
                tokens = line.split()
                tmp_meta = pos_sub_metadata[pos_sub_metadata['sent_index'] == sent_index]
                colors = ast.literal_eval(tmp_meta['color'].iloc[0])
                colors_indexes = ast.literal_eval(tmp_meta['word_index'].iloc[0])
                for k in range(len(colors)):
                    color_to_insert = colors[k]
                    index_to_insert = colors_indexes[k]
                    new_line = tokens.copy()
                    new_line.insert(index_to_insert, color_to_insert)
                    final_sent = ' '.join(new_line) + '/n'

            elif sent_index in neg_indexes:
                tokens = line.split()
                tmp_meta = neg_sub_metadata[neg_sub_metadata['sent_index'] == sent_index]
                colors = ast.literal_eval(tmp_meta['color'].iloc[0])
                colors_indexes = ast.literal_eval(tmp_meta['word_index'].iloc[0])
                for k in range(len(colors)):
                    color_to_insert = colors[k]
                    index_to_insert = colors_indexes[k]
                    new_line = tokens.copy()
                    new_line.insert(index_to_insert, color_to_insert)
                    final_sent = ' '.join(new_line) + '/n'

            else:
                final_sent = line

            tmp_file = f"tmp_dataset/{target}_{ratio}"
            final_file = f"learnt_embeddings/{target}_{ratio}"

            with open(f"{tmp_file}.txt", "a") as f:
                f.write(final_sent)

    os.remove(copied_to)
    print("Starts training corpus")
    train_corpus(tmp_file, final_file)


if __name__ == "__main__":
    start_time = time.time()
    # approximately 3 hours
    # 12k black and positive sentence is available

    input_color = 'black'
    pos_attributes_path = "is_black-pos"
    neg_attributes_path = "is_black-neg-all"

    metadata_path = "../remove_attributes/modified_dataset/remove_target_color_black_metadata.csv"
    data_path = "../remove_attributes/modified_dataset/remove_target_color_black"

    input_color = 'yellow'
    pos_attributes_path = "is_yellow-pos"
    neg_attributes_path = "is_yellow-neg-all"

    metadata_path = "../remove_attributes/modified_dataset/remove_target_color_yellow_metadata.csv"
    data_path = "../remove_attributes/modified_dataset/remove_target_color_yellow"


    # variations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    # var = [4000, 8000]
    proportion = [0.5]

    # Add all half pos and half neg back to the sentence

    for v in proportion:
        run_experiment(input_color, v, metadata_path)

    print("--- %s seconds ---" % (time.time() - start_time))
