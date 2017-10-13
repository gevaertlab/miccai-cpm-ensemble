from build_utils import *

import numpy as np

################################################################################
# fine tune rembrandt
################################################################################

test_dict = {
             'res_path': '',
             'ckpt_path': './brats_pretrained/brats.ckpt',
             'train_path': '',
             'val_path': '',
             'lr': 1e-4,
             'l2': 1e-4,
             'batch_size': 50,
             'num_epochs': 20,
             'num_train_batches': 20,
             'num_val_batches': 20,
             }

fracs = [0.2]
# fracs = [0.5]

for training_frac in fracs:

    frac_str = ''.join(str(training_frac).split('.'))

    data_path = '/share/PI/ogevaert/raghav/rembrandt/test'
    fine_tune_path = '/share/PI/ogevaert/shirley/rembrandt/fine_tune_result/fine_tune_' + frac_str

    # build_prime_dataset(data_path, fine_tune_path, training_frac)

    prime_dict = test_dict
    prime_dict['res_path'] = fine_tune_path + '/fine_tune_results.npz'
    prime_dict['train_path'] = fine_tune_path + '/train'
    prime_dict['val_path'] = fine_tune_path + '/test'
    prime_dict['fine_tune_ckpt_path'] = fine_tune_path + '/rembrandt_fine_tune.ckpt'
    prime_dict['l2'] = 1e-3
    prime_dict['num_epochs'] = 20

    build_config_file('rembrandt_fine_tune_' + frac_str, prime_dict)
    build_sbatch_file('rembrandt_fine_tune_' + frac_str, 6, 32, 'fine_tune.py')


    # build_config_file('prime_' + frac_str, prime_dict)
    # build_sbatch_file('prime_' + frac_str, 1, 4, 'prime.py')