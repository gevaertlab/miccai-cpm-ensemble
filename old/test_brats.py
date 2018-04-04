from build_utils import *

import numpy as np

################################################################################
# test on brats dataset
################################################################################

test_dict = {
		 'res_path': '/share/PI/ogevaert/shirley/brats/test_result/',
             'ckpt_path': './brats_pretrained/brats.ckpt',
             'train_path': '/share/PI/ogevaert/raghav/brats_hgg_full/train',
             'val_path': '/share/PI/ogevaert/raghav/brats_hgg_full/train',
             'lr': 1e-4,
             'l2': 1e-4,
             'batch_size': 50,
             'num_epochs': 20,
             'num_train_batches': 20,
             'num_val_batches': 20,
             }

build_config_file('brats_test', test_dict)
build_sbatch_file('brats_test', 1, 32, 'segment.py')