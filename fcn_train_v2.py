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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.full_train(sess)

        print('Saving losses and scores to %s........' %(res_path))
        np.savez(res_path,
                    train_losses=train_losses,
                    train_bdices=train_bdices,
                    val_bdices=val_bdices,
                    val_fdices=val_fdices,
                    train_ex_paths=train_ex_paths,
                    val_ex_paths=val_ex_paths,
                    config_file=config.__dict__)
