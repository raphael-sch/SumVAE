import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.python.ops.array_ops import sequence_mask

from vae.model.encoder import Encoder
from vae.model.decoder import Decoder


class SimpleVAE(object):
    def __init__(self, config, vocab_size, go_idx, eos_idx, is_training=True, ru=False):
        self.config = config
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
        self.go_idx = go_idx
        self.ru = ru
        self.is_training = is_training

        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        with tf.name_scope("Placeholders"):

            self.encoder_input = tf.placeholder(tf.int32, shape=[None, None], name="encoder_input")

            self.decoder_input = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input")

            self.target = tf.placeholder(tf.int32, shape=[None, None], name="target")

            self.input_len = tf.placeholder(tf.int32, shape=[None], name='input_len')

            self.output_len = tf.placeholder(tf.int32, shape=[None], name='output_len')

            self.step = tf.placeholder(tf.int32, shape=[], name="step")

            self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")

        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable(name="embedding",
                                             shape=[self.vocab_size, self.config.EMBED_SIZE],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.variable_scope("Encoder"):
            self.encoder = Encoder[self.config.ENCODER_NAME](self.config,
                                                             self.batch_size,
                                                             self.embedding,
                                                             self.encoder_input,
                                                             self.input_len,
                                                             is_training=self.is_training,
                                                             ru=self.ru)

        with tf.name_scope("Latent_variables"):
            if self.is_training:
                self.latent_variables = self.encoder.latent_variables
            else:
                self.latent_variables = tf.placeholder(tf.float32,
                                                       shape=(None, self.config.LATENT_VARIABLE_SIZE),
                                                       name="latent_variables_input")

        with tf.variable_scope("Decoder"):
            self.decoder = Decoder[self.config.DECODER_NAME](self.config,
                                                             self.batch_size,
                                                             self.decoder_input,
                                                             self.latent_variables,
                                                             self.embedding,
                                                             self.output_len,
                                                             self.vocab_size,
                                                             self.go_idx,
                                                             self.eos_idx,
                                                             is_training=self.is_training,
                                                             ru=self.ru)

        with tf.name_scope("Loss"):

            mask = sequence_mask(self.input_len, dtype=tf.float32)
            rnn_output = self.decoder.rnn_output
            proj_weights = self.decoder.proj_weights
            proj_bias = self.decoder.proj_bias
            self.logits = tf.tensordot(rnn_output, proj_weights, axes=((2,), (0,))) + proj_bias

            # reconstruction loss
            if self.is_training:

                def _sampled_loss(labels, logits):
                    return tf.nn.sampled_softmax_loss(tf.transpose(proj_weights),
                                                      proj_bias,
                                                      tf.expand_dims(labels, -1),
                                                      logits,
                                                      num_sampled=self.config.NUM_SAMPLED_SOFTMAX,
                                                      num_classes=vocab_size)
                reconst_loss = tf.contrib.seq2seq.sequence_loss(rnn_output, self.target, mask,
                                                                softmax_loss_function=_sampled_loss,
                                                                average_across_timesteps=False)
                self.reconst_loss = tf.reduce_sum(reconst_loss)
            else:
                # ugly filling of logits if decoding stopped to early for targets
                eos_one_hot = tf.one_hot([self.eos_idx], self.vocab_size)
                eos_seq = tf.multiply(tf.ones([self.batch_size, tf.shape(self.target)[1] - tf.shape(self.logits)[1], self.vocab_size]), eos_one_hot)
                self.logits = tf.concat([self.logits, eos_seq], axis=1)
                reconst_loss = sequence_loss(self.logits, self.target, mask, average_across_timesteps=False)
                self.reconst_loss = tf.reduce_sum(reconst_loss)

            # KLD loss
            self.kld = tf.reduce_mean(-0.5 * tf.reduce_sum(self.encoder.logvar
                                                           - tf.square(self.encoder.mu)
                                                           - tf.exp(self.encoder.logvar)
                                                           + 1, axis=1))
            self.kld_weight = tf.placeholder(tf.float32, shape=(), name="kld_weight")

            self.loss = self.reconst_loss + self.kld_weight * self.kld

            # bow loss
            if self.config.BOW_SIZE > 0:
                self.bow_logits = self.decoder.bow_logits
                bow_targets = tf.one_hot(self.target, self.vocab_size)
                bow_targets = tf.minimum(tf.reduce_sum(bow_targets, 1), 1)
                bow_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=bow_targets, logits=self.bow_logits)
                self.bow_loss = tf.reduce_mean(tf.reduce_sum(bow_loss, -1))

                self.loss += self.bow_loss

        with tf.name_scope("Summary"):
            if is_training:
                loss_summary = tf.summary.scalar("loss", self.loss, family="train_loss")
                reconst_loss_summary = tf.summary.scalar("reconst_loss", self.reconst_loss, family="train_loss")
                bow_loss_summary = tf.summary.scalar("bow_loss", self.bow_loss, family="train_loss")
                kld_summary = tf.summary.scalar("kld", self.kld, family="kld")
                kld_weight_summary = tf.summary.scalar("kld_weight", self.kld_weight, family="parameters")
                mu_summary = tf.summary.histogram("mu", tf.reduce_mean(self.encoder.mu, 0))
                var_summary = tf.summary.histogram("var", tf.reduce_mean(tf.exp(self.encoder.logvar), 0))
                lr_summary = tf.summary.scalar("lr", self.lr, family="parameters")

                self.merged_summary = tf.summary.merge([loss_summary, reconst_loss_summary, bow_loss_summary,
                                                        kld_summary, kld_weight_summary, mu_summary, var_summary,
                                                        lr_summary])

        if self.is_training:
            tvars = tf.trainable_variables()
            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.lr)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars))


VAE = {
    "SimpleVAE": SimpleVAE,
}
