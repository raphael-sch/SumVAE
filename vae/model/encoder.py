import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
import logging
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention

logger = logging.getLogger('encoder')


class EncoderVAE(object):
    def __init__(self, config, batch_size, embedding, encoder_input, input_len, is_training=True, ru=False):
        self.config = config
        with tf.variable_scope("encoder_input"):
            self.embedding = embedding
            self.encoder_input = encoder_input
            self.input_len = input_len
            self.batch_size = batch_size

            self.is_training = is_training

        with tf.variable_scope("encoder_rnn"):
            encoder_emb_inputs = tf.nn.embedding_lookup(self.embedding, self.encoder_input)

            def create_cell():
                if self.config.RNN_CELL == 'lnlstm':
                    cell = rnn.LayerNormBasicLSTMCell(self.config.ENC_RNN_SIZE)
                elif self.config.RNN_CELL == 'lstm':
                    cell = rnn.BasicLSTMCell(self.config.ENC_RNN_SIZE)
                elif self.config.RNN_CELL == 'gru':
                    cell = rnn.GRUCell(self.config.ENC_RNN_SIZE)
                else:
                    logger.error('rnn_cell {} not supported'.format(self.config.RNN_CELL))
                if self.is_training:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.DROPOUT_KEEP)
                return cell

            cell_fw = create_cell()
            cell_bw = create_cell()

            output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_emb_inputs, dtype=tf.float32)
            encoder_outputs, encoder_state = output

            def get_last_hidden():
                if self.config.RNN_CELL == 'gru':
                    return tf.concat([encoder_state[0], encoder_state[1]], -1)
                else:
                    return tf.concat([encoder_state[0][1], encoder_state[1][1]], -1)

            # last fw and bw hidden state
            if self.config.ENC_FUNC == 'mean':
                encoder_rnn_output = tf.reduce_mean(tf.concat(encoder_outputs, -1), 1)
            elif self.config.ENC_FUNC == 'lasth':
                encoder_rnn_output = get_last_hidden()
            elif self.config.ENC_FUNC in ['attn', 'attn_scale']:
                attn = seq2seq.LuongAttention(self.config.ENC_RNN_SIZE * 2,
                                              tf.concat(encoder_outputs, -1),
                                              self.input_len, scale=self.config.ENC_FUNC == 'attn_scale')
                encoder_rnn_output, _ = _compute_attention(attn, get_last_hidden(), None, None)
            elif self.config.ENC_FUNC in ['attn_ba', 'attn_ba_norm']:
                attn = seq2seq.BahdanauAttention(self.config.ENC_RNN_SIZE,
                                                 tf.concat(encoder_outputs, -1),
                                                 self.input_len, normalize=self.config.ENC_FUNC == 'attn_ba_norm')
                encoder_rnn_output, _ = _compute_attention(attn, get_last_hidden(), None, None)
            else:
                logger.error('enc_func {} not supported'.format(self.config.ENC_FUNC))

        with tf.name_scope("mu"):
            mu = tf.layers.dense(encoder_rnn_output, self.config.LATENT_VARIABLE_SIZE, activation=tf.nn.tanh)
            self.mu = tf.layers.dense(mu, self.config.LATENT_VARIABLE_SIZE, activation=None)
        with tf.name_scope("log_var"):
            logvar = tf.layers.dense(encoder_rnn_output, self.config.LATENT_VARIABLE_SIZE, activation=tf.nn.tanh)
            self.logvar = tf.layers.dense(logvar, self.config.LATENT_VARIABLE_SIZE, activation=None)

        with tf.name_scope("epsilon"):
            epsilon = tf.random_normal((self.batch_size, self.config.LATENT_VARIABLE_SIZE), mean=0.0, stddev=1.0)

        with tf.name_scope("latent_variables"):
            if self.is_training:
                self.latent_variables = self.mu + (tf.exp(0.5 * self.logvar) * epsilon)
            else:
                self.latent_variables = self.mu + (tf.exp(0.5 * self.logvar) * 0)


Encoder = {'EncoderVAE': EncoderVAE}
