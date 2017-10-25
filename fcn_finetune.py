"""Training script.

Usage:
    fcn_finetune.py (--cfg-path=<p>) [--debug]
    fcn_finetune.py -h | --help

Options:
    -h --help       Show this screen.
    --cfg-path=<p>  Config path.
    --debug         Run in debug mode.

"""
from docopt import docopt

# from models.baseline_smaller import BaselineSmallerModel
# from models.fcn import FCN_Model
# from models.fcn_new import FCNNewModel
# from models.fcn_dropout import FCNDropoutModel
# from models.fcn_bn import FCNDBnModel
# from models.fcn_bn_old import FCN_Model
from models.fcn_6 import FCN_6_Model
# from models.fcn_skip import FCN_Skip
# from models.fcn_concat import FCN_Concat

from finetune import finetune
from utils.config import Config

if __name__ == '__main__':
    arguments = docopt(__doc__)
    cfg_path = arguments['--cfg-path']
    debug = arguments['--debug']

    config = Config(cfg_path)
    model = FCN_6_Model(config)
    finetune(model, debug)
