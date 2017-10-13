"""Training script for Heidelberg dataset.

Usage:
    train_hb.py (--cfg-path=<p>) [--debug]
    train_hb.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""

from __future__ import print_function

from docopt import docopt

import numpy as np
import tensorflow as tf

from models.heidelberg import HeidelbergModel
from utils.config_hb import Config

def train_hb(cfg_path, debug):

    config = Config(cfg_path)

    ckpt_path = config.ckpt_path
    res_path = config.res_path
    
    model = HeidelbergModel(config)

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
    
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
            
            print('\nvalidate\n')
            ex_bdices = []
            for ex, ex_path in enumerate(val_ex_paths):
                print(ex)
                bdices = model._validate(ex_path, sess)
                ex_bdices.append(np.mean(bdices))
            val_bdices.append(np.mean(ex_bdices))
            
            if epoch % 5 == 4:
                print('\nsegment\n')
                ex_fdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    print(ex)
                    _, _, _, fdice = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)
                if np.mean(ex_fdices) >= best_fdice:
                    best_fdice = np.mean(ex_fdices)
                    saver.save(sess, ckpt_path)
                val_fdices.append(np.mean(ex_fdices))

        np.savez(res_path,
                 train_losses=train_losses,
                 train_bdices=train_bdices,
                 val_bdices=val_bdices,
                 val_fdices=val_fdices,
                 train_ex_paths=train_ex_paths,
                 val_ex_paths=val_ex_paths)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train_hb(arguments['--cfg-path'], arguments['--debug'])
