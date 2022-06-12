import glob
import os
import pandas as pd
from sklearn import metrics
import pathlib
from pathlib import Path
pathlib.Path(__file__).parent.resolve()


def load_results(path):
    with open(path) as infile:
        lines = infile.read().strip().split('\n')
    predictions = [line.split(',')[1] for line in lines]
    return predictions


def evaluate(labels, predictions):
    clean_labels = []
    clean_predictions = []

    if len(labels) == len(predictions):
        oov = 0
        for i, pred in enumerate(predictions):
            if pred != 'OOV':
                clean_predictions.append(int(pred))
                clean_labels.append(labels[i])
            else:
                oov += 1
        f1 = metrics.f1_score(clean_labels, clean_predictions)
        precision = metrics.precision_score(clean_labels, clean_predictions)
        recall = metrics.recall_score(clean_labels, clean_predictions)

    else:
        f1 = -1
        precision = -1
        recall = -1
        oov = -1

    return f1, precision, recall, oov


def load_gold(feature):
    data = '../test_data/'

    with open(f"{data}{feature}-pos.txt") as infile:
        words_pos = infile.read().strip().split('\n')

    with open(data + feature + '-neg-all.txt') as infile:
        words_neg = infile.read().strip().split('\n')

    return words_pos, words_neg


def get_truth(feature):
    words_pos, words_neg = load_gold(feature)
    labels = [1 for word in words_pos]
    [labels.append(0) for word in words_neg]
    return labels


def evaluate_all(feature):
    labels = get_truth(feature)
    scores = []
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results_paths = glob.glob(f'../results/*/*/{feature}.txt', recursive=True)
    cols = ['f1', 'p', 'r', 'oov']
    indices = []
    values = []

    for path in results_paths:
        predictions = load_results(path)
        name = '-'.join(path.strip('.').split('.')[0].split('/')[1:])
        f1, precision, recall, oov = evaluate(labels, predictions)
        indices.append(name)
        values.append((f1, precision, recall, oov))

    df = pd.DataFrame(values, columns=cols, index=indices).sort_values('f1', ascending=False)

    return df


def main():
    dir = '../evaluation/'
    features = ['is_black']

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for feat in features:
        print(feat)

        scores = evaluate_all(feat)
        scores.to_csv(dir + feat + '.csv')


if __name__ == "__main__":
    main()
