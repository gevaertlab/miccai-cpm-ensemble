"""Priming script.

Usage:
    prime.py (--cfg-path=<p>) [--debug]
    prime.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""

from __future__ import print_function

import os

from docopt import docopt

import numpy as np
import tensorflow as tf

from models.baseline_smaller import BaselineSmallerModel
from utils.config import Config

def prime(cfg_path, debug):

    # rembrandt_ckpt_path = '/share/PI/ogevaert/shirley/rembrandt/fine_tune_result/fine_tune_02/rembrandt_fine_tune_lr.ckpt'

    config = Config(cfg_path)
    
    ckpt_path = config.ckpt_path
    res_path = config.res_path
    
    model = BaselineSmallerModel(config)

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, ckpt_path)

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
        best_fdice = 0
    
        for epoch in range(model.config.num_epochs):
            print('\nepoch {}'.format(epoch))

            print('\nfine_tuning\n')
            ex_bdices = []
            for ex, ex_path in enumerate(train_ex_paths):
                # print(ex_path)
                losses, bdices = model._train(ex_path, sess)
                train_losses.extend(losses)
                ex_bdices.append(np.mean(bdices))
                print('******* Epoch %d Example %d: Training loss %5f' %(epoch, ex, np.mean(losses)))

            train_bdices.append(np.mean(ex_bdices))            
            print('******************** Epoch %d: Training dice score %5f' %(epoch, np.mean(ex_bdices)))


            if epoch % 5 == 0:
                print('\nvalidate\n')
                ex_bdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    bdices = model._validate(ex_path, sess)
                    ex_bdices.append(np.mean(bdices))
                    print('******* Epoch %d Example %d: Validation accuracy %5f' %(epoch, ex, np.mean(bdices)))

                val_bdices.append(np.mean(ex_bdices))
                print('******************** Epoch %d: Validation dice score %5f' %(epoch, np.mean(ex_bdices)))


                print('\ntest\n')
                ex_fdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    _, _, _, fdice = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)
                    print('******* Epoch %d Example %d: Test accuracy %5f' %(epoch, ex, fdice))

                if np.mean(ex_fdices) >= best_fdice:
                    best_fdice = np.mean(ex_fdices)
                    saver.save(sess, config.fine_tune_ckpt_path)
                    
                val_fdices.append(np.mean(ex_fdices))
                print('******************** Epoch %d: Test dice score %5f' %(epoch, np.mean(ex_fdices)))


                # print('\ntest\n')
                # val_fdices = []
                # for ex, ex_path in enumerate(val_ex_paths):
                #     print(ex)
                #     fy, fpred, _, fdice = model._segment(ex_path, sess)
                #     np.savez(os.path.join(ex_path, 'pred.npz'), fpred=fpred)
                #     np.savez(os.path.join(ex_path, 'y.npz'), fy=fy)
                #     val_fdices.append(fdice)

            
                print('\n################################################### saving checkpoint results to %s \n' %(res_path))
                np.savez(res_path,
                         train_losses=train_losses,
                         train_bdices=train_bdices,
                         val_bdices=val_bdices,
                         val_fdices=val_fdices,
                         train_ex_paths=train_ex_paths,
                         val_ex_paths=val_ex_paths)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    prime(arguments['--cfg-path'], arguments['--debug'])
