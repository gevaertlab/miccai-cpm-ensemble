from build_utils import *

import numpy as np

################################################################################
# test on rembrandt dataset
################################################################################

test_dict = {
	       'res_path': '/share/PI/ogevaert/shirley/rembrandt/fine_tune_result/fine_tune_02/eval/',
             'ckpt_path': '/share/PI/ogevaert/shirley/rembrandt/fine_tune_result/fine_tune_02/rembrandt_fine_tune.ckpt',
             'fine_tune_ckpt_path': '_',
             'train_path': '_',
             'val_path': '/share/PI/ogevaert/shirley/rembrandt/fine_tune_result/fine_tune_02/test/',
             'lr': 1e-4,
             'l2': 1e-4,
             'batch_size': 50,
             'num_epochs': 20,
             'num_train_batches': 20,
             'num_val_batches': 20,
             }

build_config_file('rembrandt_test', test_dict)
build_sbatch_file('rembrandt_test', 1, 32, 'segment.py')