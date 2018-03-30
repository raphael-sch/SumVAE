import logging
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from vae.utils.tf_utils import AlignmentWrapper, LenControlWrapper

logger = logging.getLogger('decoder')


class DecoderVAE(object):
    def __init__(self, config, batch_size, decoder_input, latent_variables, embedding, output_len, vocab_size, go_idx, eos_idx, is_training=True, ru=False):
        self.config = config
        with tf.name_scope("decoder_input"):
            self.batch_size = batch_size
            self.decoder_input = decoder_input
            self.latent_variables = latent_variables
            self.embedding = embedding
            self.output_len = output_len
            self.vocab_size = vocab_size
            self.go_idx = go_idx
            self.eos_idx = eos_idx
            self.is_training = is_training

        with tf.variable_scope("Length_Control"):
            if self.config.LEN_EMB_SIZE > 0:
                self.len_embeddings = tf.get_variable(name="len_embeddings",
                                                      shape=(self.config.NUM_LEN_EMB, self.config.LEN_EMB_SIZE),
                                                      dtype=tf.float32,
                                                      initializer=tf.random_normal_initializer(stddev=0.1))

        def create_cell():
            if self.config.RNN_CELL == 'lnlstm':
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.config.DEC_RNN_SIZE)
            elif self.config.RNN_CELL == 'lstm':
                cell = tf.contrib.rnn.BasicLSTMCell(self.config.DEC_RNN_SIZE)
            elif self.config.RNN_CELL == 'gru':
                cell = tf.contrib.rnn.GRUCell(self.config.DEC_RNN_SIZE)
            else:
                logger.error('rnn_cell {} not supported'.format(self.config.RNN_CELL))
            if self.is_training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.DROPOUT_KEEP)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([create_cell() for _ in range(2)])

        projection_layer = Dense(self.vocab_size)
        projection_layer.build(self.config.DEC_RNN_SIZE)
        self.beam_ids = self.get_beam_ids(cell, projection_layer)

        if self.config.LEN_EMB_SIZE > 0:
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            cell = LenControlWrapper(cell, self.output_len, self.len_embeddings, initial_cell_state=initial_state)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        cell = AlignmentWrapper(cell, latent_variables, initial_cell_state=initial_state)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        if self.is_training:
            decoder_emb_inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
            helper = seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_inputs, self.output_len, self.embedding, self.config.SAMP_PROB)
        else:
            helper = seq2seq.GreedyEmbeddingHelper(self.embedding, self.go_input(), self.eos_idx)

        decoder = seq2seq.BasicDecoder(cell,
                                       helper,
                                       initial_state=initial_state,
                                       output_layer=None)
        outputs, _, seq_len = seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.output_len))
        self.rnn_output = outputs.rnn_output
        self.proj_weights = projection_layer.kernel
        self.proj_bias = projection_layer.bias

        bow_h = tf.layers.dense(self.latent_variables, self.config.BOW_SIZE, activation=tf.tanh)
        if self.is_training:
            bow_h = tf.nn.dropout(bow_h, self.config.DROPOUT_KEEP)

        self.bow_logits = tf.layers.dense(bow_h, self.vocab_size, name="bow_logits")

    def go_input(self):
        go_input = tf.tile([self.go_idx], [self.batch_size])
        return go_input

    def get_beam_ids(self, cell, projection_layer):
        initial_state = cell.zero_state(self.batch_size * self.config.BEAM_WIDTH, dtype=tf.float32)

        if self.config.LEN_EMB_SIZE > 0:
            output_seq_len = seq2seq.tile_batch(self.output_len, multiplier=self.config.BEAM_WIDTH)
            cell = LenControlWrapper(cell, output_seq_len, self.len_embeddings, initial_cell_state=initial_state)
            initial_state = cell.zero_state(self.batch_size * self.config.BEAM_WIDTH, dtype=tf.float32)

        latent_variables = seq2seq.tile_batch(self.latent_variables, multiplier=self.config.BEAM_WIDTH)
        cell = AlignmentWrapper(cell, latent_variables, initial_cell_state=initial_state)
        initial_state = cell.zero_state(self.batch_size * self.config.BEAM_WIDTH, dtype=tf.float32)

        if not self.is_training:
            decoder = seq2seq.BeamSearchDecoder(cell, self.embedding, self.go_input(), self.eos_idx,
                                                initial_state=initial_state,
                                                beam_width=self.config.BEAM_WIDTH,
                                                output_layer=projection_layer)
            outputs, _, seq_len = seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.output_len))
            return outputs.predicted_ids[:, :, 0]


Decoder = {'DecoderVAE': DecoderVAE}
