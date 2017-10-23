"""Priming script.

Usage:
    prime.py (--cfg-path=<p>) [--debug]
    prime.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""
from docopt import docopt

# from models.baseline_smaller import BaselineSmallerModel
from models.cnn_only_bn import BaselineOnlyBnModel
# from models.baseline_dropout import BaselineDropoutModel

from test import test
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']
    debug = arguments['--debug']

    config = Config(cfg_path)
    model = BaselineOnlyBnModel(config)
    test(model, debug)