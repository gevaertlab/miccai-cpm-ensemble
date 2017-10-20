"""Training script for Heidelberg dataset.

Usage:
    train_hb.py (--cfg-path=<p>) [--debug]
    train_hb.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""
from docopt import docopt

from models.heidelberg import HeidelbergModel
from train import train
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']

    config = Config(cfg_path)
    model = HeidelbergModel(config)
    train(model, debug)
