"""Training script.

Usage:
    train.py (--cfg-path=<p>) [--debug]
    train.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""

from __future__ import print_function

from docopt import docopt

import numpy as np
import tensorflow as tf

# from models.baseline_bn import BaselineBnModel
from models.cnn_only_bn import BaselineOnlyBnModel
# from models.baseline_smaller import BaselineSmallerModel
# from models.baseline_dropout import BaselineDropoutModel

from utils.config import Config
from utils.general import Progbar


def train(cfg_path, debug):

    config = Config(cfg_path)

    ckpt_path = config.ckpt_path
    res_path = config.res_path
    
    model = BaselineOnlyBnModel(config)

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    prog = Progbar(target=len(train_ex_paths))
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer()) 

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
        best_fdice = 0


        print('Initialization......')
        print('validate')
        ex_bdices = []
        for ex, ex_path in enumerate(val_ex_paths):
            # print(ex_path)
            bdices = model._validate(ex_path, sess)
            ex_bdices.append(np.mean(bdices))
            # print('******* Epoch %d Example %d: Validation accuracy %5f' %(0, ex, np.mean(bdices)))   # average dice score for 20 batches of every sample  

        val_bdices.append(np.mean(ex_bdices))
        print('******************** Epoch %d: Validation dice score %5f' %(0, np.mean(ex_bdices)))     # average dice score for all samples


        print('test')
        ex_fdices = []
        for ex, ex_path in enumerate(val_ex_paths):
            # print(ex_path)
            _, _, _, fdice = model._segment(ex_path, sess)
            ex_fdices.append(fdice)
            # print('******* Epoch %d Example %d: Test accuracy %5f' %(0, ex, fdice))             # full dice score per sample

        if np.mean(ex_fdices) >= best_fdice:
            best_fdice = np.mean(ex_fdices)
            saver.save(sess, ckpt_path)
            
        val_fdices.append(np.mean(ex_fdices))
        print('******************** Epoch %d: Test dice score %5f' %(0, np.mean(ex_fdices)))        # average full dice score for all samples

        for epoch in range(1, model.config.num_epochs+1):
            print('\nepoch {}'.format(epoch))

            print('train')
            ex_bdices = []
            for ex, ex_path in enumerate(train_ex_paths):
                # print(ex_path)
                losses, bdices = model._train(ex_path, sess)
                train_losses.extend(losses)
                ex_bdices.append(np.mean(bdices))

                # logging
                prog.update(ex + 1, values=[('loss', np.mean(losses)), ('dice_score', np.mean(bdices))])
                # print('******* Epoch %d Example %d: Training loss %5f' %(epoch, ex, np.mean(losses)))   # average loss for 20 batches of every sample

            train_bdices.append(np.mean(ex_bdices))
            print('******************** Epoch %d: Training dice score %5f' %(epoch, np.mean(ex_bdices)))    # average dice score for all samples


            if epoch % 3 == 0:
                print('validate')
                ex_bdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    bdices = model._validate(ex_path, sess)
                    ex_bdices.append(np.mean(bdices))
                    # print('******* Epoch %d Example %d: Validation accuracy %5f' %(epoch, ex, np.mean(bdices)))   # average dice score for 20 batches of every sample  

                val_bdices.append(np.mean(ex_bdices))
                print('******************** Epoch %d: Validation dice score %5f' %(epoch, np.mean(ex_bdices)))     # average dice score for all samples


                print('test')
                ex_fdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    _, _, _, fdice = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)
                    # print('******* Epoch %d Example %d: Test accuracy %5f' %(epoch, ex, fdice))             # full dice score per sample

                if np.mean(ex_fdices) >= best_fdice:
                    best_fdice = np.mean(ex_fdices)
                    saver.save(sess, ckpt_path)
                    print('Saving checkpoint to %s ......' %(ckpt_path))
                    
                val_fdices.append(np.mean(ex_fdices))
                print('******************** Epoch %d: Test dice score %5f' %(epoch, np.mean(ex_fdices)))        # average full dice score for all samples


            print('Saving training losses to %s........' %(res_path))
            np.savez(res_path,
                 train_losses=train_losses,
                 train_bdices=train_bdices,
                 val_bdices=val_bdices,
                 val_fdices=val_fdices,
                 train_ex_paths=train_ex_paths,
                 val_ex_paths=val_ex_paths)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train(arguments['--cfg-path'], arguments['--debug'])
