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

# from models.baseline_smaller import BaselineSmallerModel
from models.fcn_bn import FCN_Model
from utils.config import Config


patch = 24
def eval(cfg_path, debug):

    config = Config(cfg_path)
    
    ckpt_path = config.ckpt_path
    res_path = config.res_path
    
    model = FCN_Model(config)

    val_ex_paths = model.val_ex_paths
    if debug:
        val_ex_paths = val_ex_paths[:2]

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, ckpt_path)


        print('\ntest\n')
        ex_fdices = []

        fdices = []
        for ex, ex_path in enumerate(val_ex_paths):
            print(ex, ex_path)

            fy, fpred, fprob, fdice = model._segment(ex_path, sess)

            fy = np.array(fy)
            fpred = np.array(fpred)
            fprob = np.array(fprob)
            fdice = np.array(fdice)

            ##########  example id for rembrandt 
            ex_id = ex_path[-7:-1]
            ##########  example id for brats
            # ex_id = ex_path.split('_')[-2] + '_' + ex_path.split('_')[-1]

            ex_result_path = res_path + ex_id + '.npz'
            print('saving test result to %s' %(ex_result_path))
            print(fdice)

            np.savez(ex_result_path,
                     y = fy,
                     pred = fpred,
                     prob = fprob,
                     dice = fdice
                     )

            fdices.append(fdice)

        dice_result_path = res_path + 'dice_results.npz'
        print('saving all dice scores to %s' %(dice_result_path))
        fdices = np.array(fdices)
        print('average dice scores: %f' %(np.mean(fdices)))
        np.savez(dice_result_path,
                 dices=fdices,
                 val_ex_paths=val_ex_paths)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    eval(arguments['--cfg-path'], arguments['--debug'])