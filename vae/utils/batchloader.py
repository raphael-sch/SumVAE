import numpy as np
from random import shuffle
import logging

logger = logging.getLogger('batchloader')


class BatchLoader:

    def __init__(self, vocab_file, vocab_size, batch_size=None):
        """
        Stores input data and vocabulary information. Yields training data batch_wise, sorted into buckets with equally
        sized sentences.
        :param vocab_file: path to vocabulary file
        :param vocab_size: use top x words
        :param batch_size: batch size to yield data
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.go_token = '<GO>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        self.word_to_idx, self.idx_to_word = self.get_vocabs(vocab_file)
        self.vocab_size = len(self.word_to_idx)
        logger.info('Vocab size: {}'.format(self.vocab_size))

        self.idx_to_word[self.word_to_idx[self.unk_token]] = 'UNK'

        self.eos_idx = self.word_to_idx[self.eos_token]
        self.go_idx = self.word_to_idx[self.go_token]
        self.unk_idx = self.word_to_idx[self.unk_token]

        self.data = None
        self.data_len = None
        self.steps_per_epoch = None
        self.output_order = None

    def get_data(self):
        """
        Gets data out of buckets.
        :return: one list of input data
        """
        # return list of sentence string by order
        return [' '.join([self.idx_to_word[i] for i in d]) for b in self.data for d in b]

    def get_vocabs(self, vocab_file):
        """
        Create top x words vocab and word indices from given vocab file.
        :param vocab_file: path to vocabulary file
        :return: (dict) word to idx, (dict) idx to word
        """
        word_to_idx = {self.go_token: 0, self.eos_token: 1, self.unk_token: 2}

        with open(vocab_file) as f:
            for line in f:
                word = line.rstrip()
                # if e.g. <UNK> already in the vocab file
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
                if len(word_to_idx) == self.vocab_size:
                    break

        idx_to_word = {idx: token for token, idx in word_to_idx.items()}
        return word_to_idx, idx_to_word

    def read_data(self, iter_data, max_len=None, buckets=list(range(10, 101, 10))):
        """
        Read input data into buckets. Buckets are collections of sentences with roughly the same length.
        This speeds up training as encoding and decoding steps are determined by the longest sequence in batch.
        :param iter_data: iterable of input sentences
        :param max_len: skip sentences longer than this
        :param buckets: list of bucket thresholds
        :return: the order of the data as they appear in the input iterable
        """
        if buckets[-1] != -1:
            buckets.append(-1)
        if type(iter_data) == str:
            raise ValueError('iter_data need to be opened file or list of strings')
        data = [[] for _ in buckets]
        order = [[] for _ in buckets]
        for line_idx, line in enumerate(iter_data):
            words = line.rstrip().split(' ')
            if max_len is not None and len(words) > max_len:
                continue
            word_idxs = [self.word_to_idx.get(word, self.unk_idx) for word in words]
            for b_idx, bucket in enumerate(buckets):
                if len(word_idxs) <= bucket or bucket == -1:
                    data[b_idx].append(word_idxs)
                    order[b_idx].append(line_idx)
                    break
        self.data = data
        self.data_len = sum([len(d) for d in data])
        self.steps_per_epoch = int(self.data_len / self.batch_size)
        self.output_order = [line_idx for b in order for line_idx in b]
        logger.debug('steps per epoch: {}'.format(self.steps_per_epoch))
        logger.info('instances: {}'.format(self.data_len))
        return self.output_order

    def next_batch(self, do_shuffle=True, dropword_keep=1.0):
        """
        Yield input data batch wise. Provide encoder/decoder sequence and sequence lengths
        :param do_shuffle: shuffle data
        :param dropword_keep: drop words in the decoder sequence
        :return: encoder_input, decoder_input, decoder_target, seq_lengths, batch_size
        """
        indexes = list()
        if do_shuffle:
            for i, data_bucket in enumerate(self.data):
                r = list(range(len(data_bucket)))
                shuffle(r)
                chunks = self.get_chunks(r)
                for chunk in chunks:
                    indexes.append((i, chunk))
            shuffle(indexes)
        else:
            for i, data_bucket in enumerate(self.data):
                chunks = self.get_chunks(range(len(data_bucket)))
                for chunk in chunks:
                    indexes.append((i, chunk))

        for index_chunk in indexes:
            yield self.construct_tensors(self.data, index_chunk, dropword_keep)

    def construct_tensors(self, data, index_chunk, dropword_keep):
        """
        Construct numpy arrays to be fed into tensorflow model.
        :param data: input data
        :param index_chunk: indexes of data to be loaded in current batch
        :param dropword_keep: drop words in the decoder sequence
        :return: encoder_input, decoder_input, decoder_target, seq_lengths, batch_size
        """
        i, chunk = index_chunk
        batch_data = [list(data[i][c]) for c in chunk]

        batch_size = len(batch_data)
        seq_lengths = [len(seq) for seq in batch_data]
        max_len = max(seq_lengths)

        encoder_input = np.full((batch_size, max_len+1), self.eos_idx, dtype=np.int32)
        decoder_input = np.full((batch_size, max_len+1), self.eos_idx, dtype=np.int32)

        for i, sequence in enumerate(batch_data):
            encoder_input[i, :len(sequence)] = sequence
            
            decoder_input_seq = sequence
            if dropword_keep < 1.0:
                r = np.random.rand(max_len)
                for j, idx in enumerate(sequence):
                    if r[j] > dropword_keep:
                        decoder_input_seq[j] = self.unk_idx
            decoder_input_seq = [self.go_idx] + decoder_input_seq
            decoder_input[i, :len(decoder_input_seq)] = decoder_input_seq

        decoder_target = np.copy(encoder_input)
        seq_lengths = np.array(seq_lengths) + 1
        return [encoder_input, decoder_input, decoder_target, seq_lengths, batch_size]

    def logits2str(self, logits, sample_num=1, onehot=True, stop_eos=True, postprocess=False):
        """
        Translates a list of word idx into a sentence.
        :param logits: logit output of model
        :param sample_num: return this many translated sentences
        :param onehot: logits are one hot or not
        :param stop_eos: stop translating if see EOS tag
        :param postprocess: replace special symbol literals
        :return: human readable sentence
        """

        def postprocessing(words):
            # replace normalized symbols
            replace = {'``': '"', "''": '"', '-rrb-': ')', '-lrb-': '('}
            text = ''
            # copy numbers from input to output if ## has same length
            for i, word in enumerate(words[:]):
                # remove spaces before . , etc.
                if any(word.startswith(c) for c in ["''", "'s", '-rrb-', ',', '.', ';', ':', "'t"]):
                    text = text[:-1] + replace.get(word, word) + ' '
                else:
                    text += replace.get(word, word)
                    if word not in ['``', '-lrb-']:
                        text += ' '
            # remove last space
            return text if text[-1] != ' ' else text[:-1]

        generated_texts = []
        if sample_num < 1:
            sample_num = len(logits)

        indices = logits[:sample_num]
        if onehot:
            indices = np.argmax(logits, -1)

        for i in range(sample_num):
            s = list()
            for word_idx in indices[i]:
                if stop_eos and word_idx == self.eos_idx:
                    break
                s.append(self.idx_to_word[word_idx])
            if postprocess:
                text = postprocessing(s)
            else:
                text = ' '.join(s)
            generated_texts.append(text)

        return generated_texts

    def get_chunks(self, list):
        """
        Split input list in batch_sized sized chunks.
        :param list: input data
        :return: list of chunks
        """
        return [list[offs:offs + self.batch_size] for offs in range(0, len(list), self.batch_size)]

