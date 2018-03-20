"""Training script.

Usage:
    fcn_train.py (--cfg-path=<p>)
    fcn_train.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
"""
from docopt import docopt

import tensorflow as tf

from models.fcn_concat import FCN_Concat
from models.fcn_concat_v2 import FCN_Concat_v2
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']

    config = Config(cfg_path)
    # model = FCN_Concat(config)
    model = FCN_Concat_v2(config)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        model.finetune(sess)
