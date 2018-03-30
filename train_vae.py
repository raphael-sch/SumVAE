import os
import logging
import random

import math
import numpy as np
import tensorflow as tf

from vae.model.vae import VAE
from config import get_config
from vae.utils.batchloader import BatchLoader
import summarize
from sampling import SessionSampler

logger = logging.getLogger('train')


def main(config):
    """
    Main training function. Creates train and test VAE model. Reads through input data with batchloader and trains model batch wise.
    Evaluate on dev data every x epochs and write learned weights to output.
    :param config: hyperparameter read from yaml file
    :return: trained model
    """
    random.seed(config.RANDON_SEED)
    model_dir = os.path.join(config.OUTPUT_DIR, 'model')
    batchloader_train = BatchLoader(config.VOCAB_FILE, config.VOCAB_SIZE, config.BATCH_SIZE)
    batchloader_train.read_data(open(config.TRAIN_FILE), max_len=config.MAX_SENTENCE_LENGTH)
    batchloader_dev = BatchLoader(config.VOCAB_FILE, config.VOCAB_SIZE, batch_size=32)

    with tf.Graph().as_default():
        with tf.Session(config=config.SESS_CONFIG) as sess:

            with tf.variable_scope('VAE'):
                vae = VAE[config.VAE_NAME](config,
                                           batchloader_train.vocab_size,
                                           batchloader_train.go_idx,
                                           batchloader_train.eos_idx, is_training=True, ru=False)
            with tf.variable_scope('VAE', reuse=True):
                vae_test = VAE[config.VAE_NAME](config,
                                                batchloader_dev.vocab_size,
                                                batchloader_dev.go_idx,
                                                batchloader_dev.eos_idx, is_training=False, ru=True)

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            logger.info('start training')

            reconst_loss_sum, bow_loss_sum, kld_sum, loss_sum = [], [], [], []

            step = 0
            total_steps = config.EPOCHS * batchloader_train.steps_per_epoch
            for epoch in range(config.EPOCHS):
                logger.info('epoch {}'.format(epoch+1))
                for batch_num, batch in enumerate(batchloader_train.next_batch(dropword_keep=config.DROPWORD_KEEP)):
                    lr = config.LEARNING_RATE * (1.1 - (step / (1.0 * total_steps)))
                    kld_weight = get_kl_weight(step, config, batchloader_train, config.EPOCHS)
                    encoder_input, decoder_input, target, seq_len, batch_size = batch
                    step += 1

                    feed_dict = {vae.encoder_input: encoder_input,
                                 vae.decoder_input: decoder_input,
                                 vae.target: target,
                                 vae.input_len: seq_len,
                                 vae.output_len: seq_len,
                                 vae.kld_weight: kld_weight,
                                 vae.step: step,
                                 vae.lr: lr,
                                 vae.batch_size: batch_size}

                    loss, reconst_loss, bow_loss, kld, merged_summary, _ = sess.run([vae.loss,
                                                                                     vae.reconst_loss,
                                                                                     vae.bow_loss,
                                                                                     vae.kld,
                                                                                     vae.merged_summary,
                                                                                     vae.train_op],
                                                                                    feed_dict=feed_dict)
                    reconst_loss_sum.append(reconst_loss)
                    bow_loss_sum.append(bow_loss)
                    kld_sum.append(kld)
                    loss_sum.append(loss)
                    if step % 5 == 0:
                        summary_writer.add_summary(merged_summary, step)

                if (epoch+1) % 5 == 0:
                    logger.info('epoch {} batch {} step {}'.format(epoch + 1, batch_num + 1, step))

                    logger.info('    loss: {0:.5f}'.format(np.average(loss_sum)))
                    logger.info('    reconst_loss: {0:.5f}'.format(np.average(reconst_loss_sum)))
                    logger.info('    bow_loss: {0:.5f}'.format(np.average(bow_loss_sum)))
                    logger.info('    kld {0:.5f}'.format(np.average(kld_sum)))
                    logger.info('    kld weight {0:.5f}'.format(kld_weight))

                    loss_sum, reconst_loss_sum, kld_sum = [], [], []

                    encoder_input_texts = batchloader_train.logits2str(encoder_input, onehot=False)
                    logger.info('    train input: {}'.format(encoder_input_texts[0]))
                    decoder_input_texts = batchloader_train.logits2str(decoder_input, onehot=False)
                    logger.info('    decoder input: {}'.format(decoder_input_texts[0]))
                    sample_train_outputs = batchloader_train.logits2str(sess.run(vae.logits, feed_dict))
                    logger.info('    train output: {}'.format(sample_train_outputs[0]))

                    run_valid(sess, vae_test, batchloader_dev, config.DEV_DIR, step, kld_weight, summary_writer)

                    # save model
                    save_path = saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
                    logger.info('Model saved in file {}'.format(save_path))
            run_valid(sess, vae_test, batchloader_dev, config.DEV_DIR, step, kld_weight, summary_writer)


def run_valid(sess, vae, batchloader, dev_dir, step, kld_weight, summary_writer):
    """
    Validate current model by using it to summarize give dev data and calculate ROUGE scores.
    :param sess: active session
    :param vae: VAE instance
    :param batchloader: batchloader instance
    :param dev_dir: Path to dev data; input.txt and references folder
    :param step: global step
    :param kld_weight: current kl term weight
    :param summary_writer: tensorflow summary writer object
    :return: score
    """

    batchloader.read_data(open(os.path.join(dev_dir, 'input.txt')))
    loss_sum, reconst_loss_sum, bow_loss_sum, kld_sum = [], [], [], []
    for batch in batchloader.next_batch(do_shuffle=False):
        encoder_input, _, target, seq_len, batch_size = batch
        feed_dict = {vae.encoder_input: encoder_input,
                     vae.target: target,
                     vae.input_len: seq_len,
                     vae.output_len: seq_len,
                     vae.kld_weight: kld_weight,
                     vae.batch_size: batch_size}
        latent_variables = sess.run(vae.encoder.latent_variables, feed_dict=feed_dict)

        feed_dict.update({vae.latent_variables: latent_variables})
        loss, reconst_loss, bow_loss, kld = sess.run([vae.loss,
                                                      vae.reconst_loss,
                                                      vae.bow_loss,
                                                      vae.kld], feed_dict=feed_dict)
        loss_sum.append(loss)
        reconst_loss_sum.append(reconst_loss)
        bow_loss_sum.append(bow_loss)
        kld_sum.append(kld)

    avg_loss = np.mean(loss_sum)
    avg_reconst_loss = np.mean(reconst_loss_sum)
    avg_bow_loss = np.mean(bow_loss_sum)
    avg_kld = np.mean(kld_sum)

    summary = tf.Summary()
    summary.value.add(tag='valid_loss', simple_value=avg_loss)
    summary.value.add(tag='valid_reconst_loss', simple_value=avg_reconst_loss)
    summary.value.add(tag='valid_bow_loss', simple_value=avg_bow_loss)
    summary.value.add(tag='valid_kld', simple_value=avg_kld)
    summary_writer.add_summary(summary, step)
    logger.info('    valid loss: {0:.5f}'.format(avg_loss))
    logger.info('    valid reconst loss: {0:.5f}'.format(avg_reconst_loss))
    logger.info('    valid bow loss: {0:.5f}'.format(avg_bow_loss))
    logger.info('    valid kld {0:.5f}'.format(avg_kld))
    # if model is trained without length control
    if config.LEN_EMB_SIZE < 0:
        return 0

    sampler = SessionSampler(sess, vae, batchloader)

    logger.info('    run summarization eval')

    scores_sum = summarize.evaluate(sampler, dev_dir, max_input_len=None, max_output_len=config.OUTPUT_LEN)
    summary.value.add(tag='sum_R1', simple_value=scores_sum['ROUGE-1'])
    summary.value.add(tag='sum_R2', simple_value=scores_sum['ROUGE-2'])
    summary.value.add(tag='sum_RL'.format(name), simple_value=scores_sum['ROUGE-L'])
    scores_long = summarize.evaluate(sampler, dev_dir, max_input_len=None, max_output_len=None)
    summary.value.add(tag='long_R1', simple_value=scores_long['ROUGE-1'])
    summary.value.add(tag='long_R2', simple_value=scores_long['ROUGE-2'])
    summary.value.add(tag='long_RL', simple_value=scores_long['ROUGE-L'])

    difference = scores_sum['ROUGE-1'] - scores_long['ROUGE-1']
    summary.value.add(tag='difference_R1', simple_value=difference)
    logger.info('    full_difference_R1: {0:.5f}'.format(difference))


def get_kl_weight(step, config, batchloader, epochs):
    """
    Anneal the KL weight based on hyperparameter mid and steep from 0 to 1 during epochs.
    :param step: current global step
    :param config: config instance
    :param batchloader: batchloader instance
    :param epochs: total epochs
    :return: kl term weight
    """
    mid_of_slope = epochs / 100.0 * batchloader.steps_per_epoch * config.KLD_MID
    steepness = epochs / 100.0 * batchloader.steps_per_epoch * config.KLD_STEEP
    return (math.tanh((step - mid_of_slope) / steepness) + 1) / 2


if __name__ == "__main__":
    config = get_config()
    main(config)
