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

from models.fcn_concat import FCN_Concat
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']
    debug = arguments['--debug']

    config = Config(cfg_path)
    model = FCN_Concat(config)

    if debug:
        model.train_ex_paths = model.train_ex_paths[:2]
        model.val_ex_paths = model.val_ex_paths[:2]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.full_train(sess)
