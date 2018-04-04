from build_utils import *

import numpy as np

################################################################################
# prime
################################################################################

train_dict = {'res_path': '/share/PI/ogevaert/train_results.npz',
             'ckpt_path': 'checkpoints/rembrandt.ckpt',
             'train_path': '/share/PI/ogevaert/brats_hgg/train',
             'test_path': '/share/PI/ogevaert/brats_hgg/val',
             'lr': 1e-4,
             'l2': 1e-4,
             'batch_size': 50,
             'num_epochs': 20,
             'num_train_batches': 20,
             'num_val_batches': 20,
             }

fracs = [0.05, 0.1, 0.15, 0.2]

build_config_file('train', train_dict)
build_sbatch_file('train', 12, 32, 'train.py')

for frac in fracs:

    frac_str = ''.join(str(frac).split('.'))

    test_path = '/share/PI/ogevaert/rembrandt/test'
    base_path = '/share/PI/ogevaert/prime_' + frac_str

    build_prime_dataset(test_path, base_path, frac)

    prime_dict = train_dict
    prime_dict['res_path'] = base_path + '_results.npz'
    prime_dict['train_path'] = base_path + '/prime'
    prime_dict['val_path'] = base_path + '/test'
    prime_dict['l2'] = 1e-3
    prime_dict['num_epochs'] = 10

    build_config_file('prime_' + frac_str, prime_dict)
    build_sbatch_file('prime_' + frac_str, 1, 4, 'prime.py')

###############################################################################
# xval
###############################################################################

# for i in range(4):
#     train_dict = {'res_path': '/share/PI/ogevaert/xval_eval_' + str(i) + '_results.npz',
#                  'ckpt_path': 'checkpoints/xval_' + str(i) + '.ckpt',
#                  'train_path': '/share/PI/ogevaert/brats_hgg_xval_' + str(i) + '/train',
#                  'val_path': '/share/PI/ogevaert/brats_hgg_xval_' + str(i) + '/val',
#                  'lr': 1e-4,
#                  'l2': 1e-4,
#                  'batch_size': 50,
#                  'num_epochs': 20,
#                  'num_train_batches': 20,
#                  'num_val_batches': 20,
#                  }
#     build_config_file('xval_eval_' + str(i), train_dict)
#     build_sbatch_file('xval_eval_' + str(i), 1, 32, 'eval.py')

###############################################################################
# heidelberg
###############################################################################

# train_dict = {'res_path': '/share/PI/ogevaert/train_hb_results.npz',
#               'ckpt_path': 'checkpoints/heidelberg.ckpt',
#               'train_path': '/share/PI/ogevaert/heidelberg/train',
#               'val_path': '/share/PI/ogevaert/heidelberg/val',
#               'lr': 1e-4,
#               'l2': 1e-4,
#               'batch_size': 50,
#               'num_epochs': 20,
#               'num_train_batches': 20,
#               'num_val_batches': 20,
#              }

# build_config_file('train_hb', train_dict)
# build_sbatch_file('train_hb', 1, 32, 'train_hb.py')

###############################################################################
# train only
###############################################################################

train_dict = {'res_path': '/share/PI/ogevaert/results/train_only_results.npz',
              'ckpt_path': 'checkpoints/brats_hgg_full.ckpt',
              'train_path': '/share/PI/ogevaert/brats_hgg_full/train',
              'val_path': '/share/PI/ogevaert/brats_hgg_full/val',
              'lr': 1e-4,
              'l2': 1e-4,
              'batch_size': 50,
              'num_epochs': 20,
              'num_train_batches': 20,
              'num_val_batches': 20,
             }

build_config_file('train_only', train_dict)
build_sbatch_file('train_only', 16, 32, 'train_only.py')
