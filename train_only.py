"""Training script (no validation).

Usage:
    train_only.py (--cfg-path=<p>) [--debug]
    train_only.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""

from __future__ import print_function

from docopt import docopt

import numpy as np
import tensorflow as tf

from models.baseline_smaller import BaselineSmallerModel
from utils.config import Config

def train_only(cfg_path, debug):

    config = Config(cfg_path)

    ckpt_path = config.ckpt_path
    res_path = config.res_path
    
    model = BaselineSmallerModel(config)

    train_ex_paths = model.train_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        train_losses = []
        train_bdices = []
    
        best_fdice = 0

        for epoch in range(model.config.num_epochs):
            print('\nepoch {}'.format(epoch))

            print('\ntrain\n')
            ex_bdices = []
            for ex, ex_path in enumerate(train_ex_paths):
                print(ex)
                losses, bdices = model._train(ex_path, sess)
                train_losses.extend(losses)
                ex_bdices.append(np.mean(bdices))
            train_bdices.append(np.mean(ex_bdices))
            
        np.savez(res_path,
                 train_losses=train_losses,
                 train_bdices=train_bdices,
                 train_ex_paths=train_ex_paths)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train_only(arguments['--cfg-path'], arguments['--debug'])
