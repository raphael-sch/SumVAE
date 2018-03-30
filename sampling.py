import tensorflow as tf
import numpy as np
import os
import sys
import logging
import argparse
import warnings

# https://github.com/tensorflow/tensorflow/issues/12927
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

logger = logging.getLogger('sampling')


class Sampler:

    def __init__(self, sess, batchloader, inputs, outputs, sampler_name=None):
        """
        Provides high level functions for a learned summarization model like sampling new sentences or pass input
        trt the VAE to generate different length outputs.
        :param sess: active session of a model
        :param batchloader: instantiated batchloader
        :param inputs: dict of input placeholders of tensorflow model
        :param outputs: dict of output placeholders of tensorflow model
        :param sampler_name: name of sampler (exp or sess)
        """
        self.sess = sess
        self.batchloader = batchloader
        self.inputs = inputs
        self.outputs = outputs
        self.sampler_name = sampler_name

    def get_latent_variables(self, data_iter, return_scales=False):
        """
        Encode given sentences into latent variables
        :param data_iter: iterable of sentences
        :param return_scales: True to return \sigma
        :return: latent_variables, (sigmas)
        """
        order = self.batchloader.read_data(data_iter, max_len=None, buckets=[10, 20, 30, 50])
        latent_variables = list()
        scales = list()
        for batch in self.batchloader.next_batch(do_shuffle=False):
            sample_input, _, sample_target, seq_len, batch_size = batch
            batch_latent_v, batch_logvar = self.sess.run([self.outputs['mu'], self.outputs['logvar']],
                                                         feed_dict={self.inputs['enc_input']: sample_input,
                                                                    self.inputs['input_len']: seq_len,
                                                                    self.inputs['batch_size']: batch_size})

            latent_variables.extend(batch_latent_v)
            batch_scales = np.exp(batch_logvar)
            scales.extend(batch_scales)

        latent_variables = _get_ordered(latent_variables, order)
        scales = _get_ordered(scales, order)
        if return_scales:
            return latent_variables, scales
        return latent_variables

    def get_text(self, latent_variables, output_len=100, postprocess=False):
        """
        Decodes sentences from given latent variables.
        :param latent_variables: list of latent variables
        :param output_len: specify output length
        :param postprocess: replace special character literals
        :return:
        """
        texts = list()
        for latent_chunk in self.batchloader.get_chunks(latent_variables):
            batch_size = len(latent_chunk)
            output_seq_len = np.full([batch_size], output_len, dtype=np.int32)
            beam_ids = self.sess.run(self.outputs['beam_ids'],
                                     feed_dict={self.inputs['latent_variables']: latent_chunk,
                                                self.inputs['batch_size']: batch_size,
                                                self.inputs['output_len']: output_seq_len})
            # somehow different behaviour if accessed from model file or active session
            if self.sampler_name == 'exp':
                # get hypotheses with highest probability
                beam_ids = beam_ids[:, :, 0]
            texts.extend(self.batchloader.logits2str(beam_ids,
                                                     sample_num=-1,
                                                     onehot=False,
                                                     postprocess=postprocess))
        return texts

    def get_resampled_text(self, data_iter, output_len=None, postprocess=False):
        """
        Reads input sentences, pass them trough VAE with possibility to change the output length.
        :param data_iter: iterable with sentences
        :param output_len: specify output length; None for same as input length
        :param postprocess: replace special character literals
        :return: resampled sentences
        """
        order = self.batchloader.read_data(data_iter, max_len=None, buckets=[10, 20, 30, 50])
        texts = list()
        for batch in self.batchloader.next_batch(do_shuffle=False, verbose=1):
            encoder_input, _, target, seq_len, batch_size = batch
            if output_len is not None:
                output_seq_len = np.full([batch_size], output_len, dtype=np.int32)
            else:
                output_seq_len = seq_len

            feed_dict = {self.inputs['enc_input']: encoder_input,
                         self.inputs['input_len']: seq_len,
                         self.inputs['output_len']: output_seq_len,
                         self.inputs['batch_size']: batch_size}
            latent_variables = self.sess.run(self.outputs['mu'], feed_dict=feed_dict)

            feed_dict.update({self.inputs['latent_variables']: latent_variables})
            beam_ids = self.sess.run(self.outputs['beam_ids'], feed_dict=feed_dict)
            # somehow different behaviour if loaded from model file or active session
            if self.sampler_name == 'exp':
                # get hypotheses with highest probability
                beam_ids = beam_ids[:, :, 0]
            texts.extend(self.batchloader.logits2str(beam_ids, sample_num=-1, onehot=False, postprocess=postprocess))
        texts = _get_ordered(texts, order)
        return texts

    def get_sampled_text(self, num=1, output_len=100):
        """
        Sample a new sentence.
        :param num: number of sampled sentences
        :param output_len: specify the length of sampled sentence
        :return: sampled latent variables, decoded sentences
        """
        latent_size = self.inputs['latent_variables'].get_shape()[1]
        latent_variables = np.random.normal(loc=0.0, scale=1.0, size=[num, latent_size])
        texts = self.get_text(latent_variables, output_len)
        return latent_variables, texts


def _get_ordered(output, order):
    """
    Reorder output of model to original order of input file.
    :param output: shuffled output of model
    :param order: original ordering indexes
    :return: ordered output
    """
    assert len(output) == len(order)
    new_output = [None for _ in output]
    for i, o in enumerate(order):
        new_output[o] = output[i]
    return new_output


class SessionSampler(Sampler):
    def __init__(self, sess, vae, batchloader):
        """
        Provides high level functions for a learned summarization model like sampling new sentences or pass input
        trt the VAE to generate different length outputs.
        :param sess: active session of a model
        :param vae: instantiated VAE model
        :param batchloader: instantiated batchloader
        """
        inputs = dict(batch_size=vae.batch_size,
                      latent_variables=vae.latent_variables,
                      input_len=vae.input_len,
                      output_len=vae.output_len,
                      enc_input=vae.encoder_input)
        outputs = dict(beam_ids=vae.decoder.beam_ids,
                       mu=vae.encoder.mu,
                       logvar=vae.encoder.logvar)
        super(SessionSampler, self).__init__(sess, batchloader, inputs, outputs)


class ExperimentSampler(Sampler):

    def __init__(self, experiment_dir, batch_size=16):
        """
        Provides high level functions for a learned summarization model like sampling new sentences or pass input
        trt the VAE to generate different length outputs. Loads an active session from a experiment folder.
        :param experiment_dir: Path to experiment directory. Should contain code, config, vocab and model subfolders.
        :param batch_size: override the batch_size from the config file
        """
        model_dir = os.path.join(experiment_dir, 'model')
        vocab_dir = os.path.join(experiment_dir, 'vocab')
        vocab_file = os.path.join(vocab_dir, os.listdir(vocab_dir)[0])
        config_dir = os.path.join(experiment_dir, 'config')
        config_file = os.path.join(config_dir, os.listdir(config_dir)[0])
        code_dir = os.path.join(experiment_dir, 'code')

        sys.path = [code_dir] + sys.path
        from config import get_config
        from vae.utils.batchloader import BatchLoader

        config = get_config(config_file)
        self.config = config
        batchloader = BatchLoader(vocab_file, config.VOCAB_SIZE, batch_size=batch_size)

        graph = tf.get_default_graph()
        sess = tf.Session(config=config.SESS_CONFIG)
        loader = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt.meta'))
        #self.sess.run(tf.global_variables_initializer())
        loader.restore(sess, tf.train.latest_checkpoint(model_dir))
        logger.info('model restored')

        # get placeholders
        batch_size = graph.get_tensor_by_name('VAE_1/Placeholders/batch_size:0')
        beam_ids = graph.get_tensor_by_name('VAE_1/Decoder/decoder/transpose:0')
        latent_variables = graph.get_tensor_by_name('VAE_1/Latent_variables/latent_variables_input:0')
        input_len = graph.get_tensor_by_name('VAE_1/Placeholders/input_len:0')
        output_len = graph.get_tensor_by_name('VAE_1/Placeholders/output_len:0')
        encoder_input = graph.get_tensor_by_name('VAE_1/Placeholders/encoder_input:0')
        mu = graph.get_tensor_by_name('VAE_1/Encoder/mu/dense_2/BiasAdd:0')
        logvar = graph.get_tensor_by_name('VAE_1/Encoder/log_var/dense_1/BiasAdd:0')

        inputs = dict(batch_size=batch_size, latent_variables=latent_variables, input_len=input_len, output_len=output_len, enc_input=encoder_input)
        outputs = dict(beam_ids=beam_ids, mu=mu, logvar=logvar)

        super(ExperimentSampler, self).__init__(sess, batchloader, inputs, outputs, sampler_name='exp')


def get_input_sentences(input_file, max_len=None):
    """
    Reads sentences line by line from given file.
    :param input_file: path to file
    :param max_len: skip sentences shorter than this
    :return: list of sentences, line number in file of sentences
    """
    sentences = list()
    indices = list()
    with open(input_file) as f:
        for i, line in enumerate(f):
            sentence = line.rstrip()
            if max_len is None or len(sentence.split(' ')) <= max_len:
                sentences.append(sentence)
                indices.append(i)
    return sentences, indices


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample new sentences or pass existing sentences through model '
                                                 'and change length')

    parser.add_argument('exp_dir', type=str,
                        help='Path to experiment directory. Should contain code, config, vocab and model subfolders.')

    parser.add_argument('task', type=str,
                        help='sample: sample new sentence; resample: read input file and pass trough model. '
                             'May choose different output length')

    parser.add_argument('-data_file', type=str, default=None,
                        help='File to read in.')

    parser.add_argument('-output_len', type=int, default=None,
                        help='Set desired length in LenEmb. Remove if whole sentence should be decoded')

    parser.add_argument('-batch_size', type=int, default=16,
                        help='Batch size. Note that actual memory is needed for batch_size x beam_width')

    args = parser.parse_args()

    sampler = ExperimentSampler(args.exp_dir)
    if args.task == 'resample':
        if args.data_file is None:
            parser.error('Must provide data_file parameter')
        input_sentences, input_indices = get_input_sentences(args.data_file, None)
        output_sentences = sampler.get_resampled_text(input_sentences, args.output_len, postprocess=False)
        for i, o in zip(input_sentences, output_sentences):
            print(i)
            print(o)
            print('')
    elif args.task == 'sample':
        if args.output_len is None:
            warnings.warn('output len must be set; fallback to 15')
            args.output_len = 15
        print(sampler.get_sampled_text(1, args.output_len)[1])
    else:
        parser.error("task parameter must be 'sample' or 'resample', not {}".format(args.task))
