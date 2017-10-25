"""Training script.

Usage:
    cnn_train.py (--cfg-path=<p>) [--debug]
    cnn_train.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.
"""

from docopt import docopt

from models.baseline_bn import BaselineBnModel
# from models.cnn_only_bn import BaselineOnlyBnModel
# from models.baseline_smaller import BaselineSmallerModel
# from models.baseline_dropout import BaselineDropoutModel

from train import train
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']
    debug = arguments['--debug']

    config = Config(cfg_path)
    model = BaselineBnModel(config)
    train(model, debug)
