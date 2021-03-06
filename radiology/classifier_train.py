"""Training script.

Usage:
    fcn_train.py (--cfg-path=<p>) [--debug]
    fcn_train.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""
from docopt import docopt

import tensorflow as tf

from radiology.models.cnn_classifier import CNN_Classifier
from radiology.utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']
    debug = arguments['--debug']

    config = Config(cfg_path)
    model = CNN_Classifier(config)

    if debug:
        model.train_ex_paths = model.train_ex_paths[:2]
        model.val_ex_paths = model.val_ex_paths[:2]

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model.full_train(sess)
