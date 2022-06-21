"""
Get the lower and upper baseline results -> prepare the emebeddings model for three scenarios:
- all target colors were removed (in a distance of +- 2 from the concept word)
- all colors from a list were removed (in a distance of +- 2 from the concept word)
- none colors were removed
"""

import time
import os
import pandas as pd
import ast
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


if __name__ == "__main__":
    train_corpus("../../raw_data/data", "no_removal_baseline")
    train_corpus("../remove_attributes/modified_dataset/remove_all_colors", "all_colors_removed")
    # train_corpus("../remove_attributes/modified_dataset/remove_target_color_black", "all_black_color_removed")
