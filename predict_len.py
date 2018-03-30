import random
from random import shuffle
import argparse
from sklearn.linear_model import LinearRegression
from sampling import ExperimentSampler

random.seed(1234)


def predict_len(sampler, input_file, test_percent=0.2):
    """
    Predict length of input sentences based on generated latent variables.
    :param sampler: sampler instance
    :param input_file: file of input sentences
    :param test_percent: train/test split
    :return:
    """
    sentences_train, sentences_test = get_data_splits(input_file, test_percent)
    x_train, y_train = get_x_y(sampler, sentences_train)
    x_test, y_test = get_x_y(sampler, sentences_test)
    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    return score


def get_x_y(sampler, sentences):
    """
    Get latent variables and construct instances for sklearn.
    :param sampler: sampler instance
    :param sentences: list of input sentences
    :return: features, labels
    """
    y = [len(s.split(' ')) for s in sentences]
    x = sampler.get_latent_variables(sentences)
    return x, y


def get_data_splits(input_file, test_percent=0.2, do_shuffle=True):
    """
    Split input sentences in train and test.
    :param input_file: path to file of input sentences
    :param test_percent: train/test split
    :param do_shuffle: shuffle input
    :return: train sentences, test sentences
    """
    sentences = list()
    with open(input_file) as f:
        for line in f:
            sentences.append(line.rstrip())
    if do_shuffle:
        shuffle(sentences)
    split_idx = int(len(sentences) * (1 - test_percent))
    sentences_train = sentences[:split_idx]
    sentences_test = sentences[split_idx:]
    print(len(sentences_train))
    print(len(sentences_test))
    return sentences_train, sentences_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict the length of given sentences from the latent variable')

    parser.add_argument('exp_dir', type=str,
                        help='Path to experiment directory. Should contain code, config, vocab and model subfolders.')

    parser.add_argument('data_file', type=str, default=None,
                        help='File to read in.')

    parser.add_argument('-batch_size', type=int, default=32,
                        help='Batch size.')

    args = parser.parse_args()

    sampler = ExperimentSampler(args.exp_dir, args.batch_size)
    score = predict_len(sampler, args.data_file, test_percent=0.2)
    print(score)
