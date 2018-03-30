import os
import shutil
import glob
from distutils.dir_util import copy_tree
import logging
import yaml
import copy
import tensorflow as tf
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('config')


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo):
        return DotDict([(copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items()])


def get_config(config_file=None):
    """
    Read hyperparameter from yaml file into config object.
    :param config_file: path to yaml file
    :return: DotDict with hyperparameter
    """

    if config_file is None:
        config_file = FLAGS.CONFIG

    f = open(config_file)
    config_yaml = yaml.safe_load(f)
    f.close()
    config_dict = dict()
    config_dict['CONFIG_FILE'] = config_file

    config_dict['EPOCHS'] = config_yaml.get('epochs', 100)
    config_dict['SAMP_PROB'] = config_yaml.get('samp_prob', 0.0)
    config_dict['DROPWORD_KEEP'] = config_yaml.get('dropword_keep', 0.62)
    config_dict['LEARNING_RATE'] = config_yaml.get('learning_rate', 0.001)
    config_dict['KLD_MID'] = config_yaml.get('kld_mid', 7)  # 5-35
    config_dict['KLD_STEEP'] = config_yaml.get('kld_steep', 2)  # 2-15
    config_dict['EMBED_SIZE'] = config_yaml.get('embed_size', 300)
    config_dict['LATENT_VARIABLE_SIZE'] = config_yaml.get('latent_variable_size', 15)
    config_dict['RNN_CELL'] = config_yaml.get('rnn_cell', 'gru').lower()
    config_dict['ENC_RNN_SIZE'] = config_yaml.get('enc_rnn_size', 128)
    config_dict['DEC_RNN_SIZE'] = config_yaml.get('dec_rnn_size', 128)
    config_dict['BOW_SIZE'] = config_yaml.get('bow_size', 300)
    config_dict['ENC_FUNC'] = config_yaml.get('enc_func', 'mean').lower()
    config_dict['VAE_NAME'] = config_yaml.get('vae_name', 'SimpleVAE')
    config_dict['ENCODER_NAME'] = config_yaml.get('encoder_name', 'EncoderVAE')
    config_dict['DECODER_NAME'] = config_yaml.get('decoder_name', 'DecoderVAE')
    config_dict['DROPOUT_KEEP'] = config_yaml.get('dropout_keep', 1.0)
    config_dict['MAX_GRAD'] = config_yaml.get('max_grad', 5.0)
    config_dict['VOCAB_SIZE'] = config_yaml.get('vocab_size', 16000)
    config_dict['NUM_SAMPLED_SOFTMAX'] = config_yaml.get('num_sampled_softmax', 100)
    config_dict['BEAM_WIDTH'] = config_yaml.get('beam_width', 10)
    config_dict['BATCH_SIZE'] = config_yaml.get('batch_size', 128)
    config_dict['MAX_SENTENCE_LENGTH'] = config_yaml.get('max_sentence_length', 20)
    config_dict['LEN_EMB_SIZE'] = config_yaml.get('len_emb_size', 50)
    config_dict['NUM_LEN_EMB'] = config_yaml.get('num_len_emb', 300)
    config_dict['OUTPUT_LEN'] = config_yaml.get('output_len', 15)
    config_dict['RANDOM_SEED'] = config_yaml.get('random_seed', 1234)

    config = DotDict(config_dict)

    config.OUTPUT_DIR = FLAGS.OUTPUT_DIR
    config.TRAIN_FILE = FLAGS.TRAIN_FILE
    config.DEV_DIR = FLAGS.DEV_DIR
    config.VOCAB_FILE = FLAGS.VOCAB_FILE
    config.SESS_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    for key, value in config.items():
        logger.info('{}: {}'.format(key, value))

    return config


flags = tf.app.flags

flags.DEFINE_string('TRAIN_FILE', "data/train/train.txt", "")
flags.DEFINE_string('DEV_DIR', "data/test/example", "")
flags.DEFINE_string('VOCAB_FILE', "data/train/vocab.txt", "")
flags.DEFINE_string('OUTPUT_DIR', "outputs/example1", "")
flags.DEFINE_string('CONFIG', "configs/example.yaml", "")

FLAGS = flags.FLAGS

# create experiment directory
code_dir = os.path.join(FLAGS.OUTPUT_DIR, 'code')
config_dir = os.path.join(FLAGS.OUTPUT_DIR, 'config')
vocab_dir = os.path.join(FLAGS.OUTPUT_DIR, 'vocab')

os.makedirs(code_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(vocab_dir, exist_ok=True)

shutil.copy2(FLAGS.VOCAB_FILE, vocab_dir)
shutil.copy2(FLAGS.CONFIG, config_dir)
copy_tree('./vae/', code_dir)
for f in glob.glob(r'./*.py'):
    shutil.copy2(f, code_dir)
