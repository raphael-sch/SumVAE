import os
import logging
import argparse
import random
from pythonrouge.pythonrouge import Pythonrouge
from sampling import ExperimentSampler, get_input_sentences
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('sum')


def evaluate(sampler, data_dir, max_input_len=20, max_output_len=50, verbose=0):
    """
    Reads input and references from data dir. Creates summaries with sampler and evaluates them with ROUGE.
    :param sampler: sampler instance
    :param data_dir: contains input.txt and references folder
    :param max_input_len: read only sentences short than this; None = read all
    :param max_output_len: specify output length; None for same as input length
    :param verbose: verbosity
    :return: ROUGE scores
    """
    logger.info('    max_input_len: {}  max_output_len: {}'.format(max_input_len, max_output_len))
    references_dir = os.path.join(data_dir, 'references')
    input_file = os.path.join(data_dir, 'input.txt')

    input_sentences, input_indices = get_input_sentences(input_file, max_input_len)
    summary_sentences = sampler.get_resampled_text(input_sentences, max_output_len, postprocess=True)

    logger.info('    instances: {}'.format(len(input_sentences)))
    r = random.randint(0, len(input_sentences)-1)
    logger.info('    input: {}'.format(input_sentences[r]))
    logger.info('    output: {}'.format(summary_sentences[r]))
    avg_sentence_len = np.mean([len(s) for s in summary_sentences])
    logger.info('    avg_chars: {}'.format(avg_sentence_len))

    summary = [[text] for text in summary_sentences]
    scores = get_rouge(input_sentences, summary, references_dir, input_indices, verbose=verbose)
    return scores


def get_rouge(input_sentences, summary, references_dir, order=None, verbose=0):
    """
    Calculate ROUGE scores for generated summaries and references.
    :param input_sentences: unmodified input sentences to calculate extractivness
    :param summary: list of summaries
    :param references_dir: contains reference files
    :param order: order during loading input
    :param verbose: verbosity
    :return: ROUGE scores
    """
    if order is None:
        order = range(len(summary))
    reference_filenames = os.listdir(references_dir)
    references_all = [[] for _ in reference_filenames]
    for i, reference_filename in enumerate(reference_filenames):
        with open(os.path.join(references_dir, reference_filename)) as f:
            for line in f:
                references_all[i].append([line.rstrip()])
    references = [[references_list[o] for references_list in references_all] for o in order]

    if verbose > 0:
        for i, s, r_list in zip(input_sentences, summary, references):
            print('input, generated sentence and references:')
            print('{}'.format(i))
            print('{}'.format(s))
            for r in r_list:
                print('{}'.format(r))
            print('')
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=references,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=True,
                        stemming=True, stopwords=False,
                        word_level=False, length_limit=True, length=75,
                        use_cf=True, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    scores = rouge.calc_score()
    logger.info('ROUGE-1: {ROUGE-1:.4f}  ROUGE-2: {ROUGE-2:.4f}  ROUGE-L: {ROUGE-L:.4f}'.format(**scores))

    words_total = 0
    words_ext = 0
    for i, s in zip(input_sentences, summary):
        words_total += len(set(s[0].split(' ')))
        words_ext += len(set(s[0].split(' ')) & set(i.split(' ')))
    logger.info('{0:.2f}% extractive'.format(words_ext / float(words_total)))

    plt.hist([len(s[0]) for s in summary], bins=30)
    plt.xlabel('output characters', fontsize=11)
    plt.savefig('plot.png')

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a summarization dataset')

    parser.add_argument('exp_dir', type=str,
                        help='Path to experiment directory. Should contain code, config, vocab and model subfolders.')

    parser.add_argument('data_dir', type=str,
                        help='Path to data directory where input.txt and references are located.')

    parser.add_argument('-batch_size', type=int, default=16,
                        help='Batch size. Note that actual memory is needed for batch_size x beam_width')

    parser.add_argument('-output_len', type=int, default=None,
                        help='Set desired length in LenEmb. Remove if whole sentence should be decoded')

    parser.add_argument('-verbose', type=int, default=1,
                        help='Verbosity: 0 or 1')

    args = parser.parse_args()

    sampler = ExperimentSampler(args.exp_dir, args.batch_size)
    logger.info("run task... this can take a few minutes depending on number of instances...")
    evaluate(sampler, args.data_dir, max_input_len=None, max_output_len=args.output_len, verbose=args.verbose)

