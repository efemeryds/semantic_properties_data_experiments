"""
Choose n sentences that had previously removed target color from the sentence and where the concept is positive.
Then input back the color to the proper location by referring to its index from metadata file.
"""


import os
import gensim
import shutil
import pandas as pd
# https://stackoverflow.com/questions/63459657/how-to-load-large-dataset-to-gensim-word2vec-model
import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(2020)


def prepare_experiments(target_name, exp_list: list, target_vocab):
    # exp talks about the amount of samples from a given target vocab
    for exp in exp_list:
        target_sample = list(np.random.choice(target_vocab, exp))
        # save target to a file

        # open the full text file and add there the sentences
        file_path = f"{target_name}_{exp}"
        for sent in target_sample:
            with open(f'../target_data/{file_path}.txt', 'a') as f:
                f.writelines(sent)
                f.close()


# make copies of neutral files and open file called as a amount of color and write there the color sentences
def run_experiments(target_file: str, exp_list: list):
    for exp in exp_list:
        dir_path = f"../full_neutral_data/"

        tmp_file = f"{exp}"
        # copy a file
        file_to_copy = f"{dir_path}{target_file}.txt"
        copied_to = f"{dir_path}{tmp_file}.txt"
        shutil.copyfile(file_to_copy, copied_to)

        # read the lines
        with open(f'../target_data/{target_file}_{exp}.txt') as f:
            lines = f.readlines()

        # add the lines
        for sent in lines:
            with open(f"{dir_path}{tmp_file}.txt", 'a') as f:
                f.writelines(sent)
                f.close()

        final_file = f"{target_file}_{exp}"
        path = f"{dir_path}{tmp_file}.txt"
        train_corpus(path, final_file)


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


def train_corpus(file_path, final_file_name: str):
    #file_path = "../../get_sentences/text_sentences/no_colors_sents.txt"
    model = Word2Vec(epochs=3, corpus_file=file_path, vector_size=100, window=5, min_count=1, workers=8, compute_loss=True,
                     callbacks=[callback()])
    model.wv.save_word2vec_format(f"../../../trained_models/gensim_text/{final_file_name}.bin", binary=True)
    print(model.wv.get_vector("black", norm=True))
    os.remove(file_path)
    print("STOP")


if __name__ == "__main__":
    feature = "black_color"
    # read data
    colors_dict = pd.read_pickle(r'../../get_sentences/pickled_sentences/colors_dictionary.pkl')
    is_black = colors_dict['black']

    # experiments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    experiments = [4, 16, 1024, 32768]
    prepare_experiments(feature, experiments, is_black)
    run_experiments("black_color", experiments)
