import numpy as np

import os

from random import shuffle

from data_utils import im_path_to_arr, arr_to_im_path

gbm_in_path = '/share/PI/ogevaert/tcga-preprocessed-images/GBM/'
              + 'TCGA_GBM_PreProc_NIFTI/'
gbm_out_path = '/share/PI/ogevaert/tcga-gbm'

lgg_in_path = '/share/PI/ogevaert/tcga-preprocessed-images/LGG/'
              + 'TCGA_GBM_PreProc_NIFTI'
lgg_out_path = '/share/PI/ogevaert/tcga-lgg'

def j(path, fname):
    return os.path.join(path, fname)

# reorganize gbm data
for ex_name in os.listdir(gbm_in_path):
    ex_path = j(gbm_in_path, ex_name)
    if os.path.isdir(ex_path):
        os.rename(j(ex_path, 'flair_t1_bcorr_brain.nii'),
                  j(ex_path, 'flair.nii'))
        os.rename(j(ex_path, 't1-post_pre_bcorr_brain.nii'),
                  j(ex_path, 't1.nii'))
        os.rename(j(ex_path, 't1-pre_bcorr_brain.nii'),
                  j(ex_path, 't1c.nii'))

# reorganize lgg data
for ex_name in os.listdir(lgg_in_path):
    ex_path = j(lgg_in_path, ex_name)
    if os.path.isdir(ex_path):
        os.rename(j(ex_path, 'flair_t1_bcorr_brain.nii'),
                  j(ex_path, 'flair.nii'))
        os.rename(j(ex_path, 't1-pre_bcorr_brain.nii'),
                  j(ex_path, 't1c.nii'))
           
# normalize data
for in_path in [gbm_in_path, lgg_in_path]:
    for ex_name in os.listdir(in_path):
        ex_path = j(in_path, ex_name)
        if os.path.isdir(ex_path):
            for im_name in os.listdir(ex_path):
                im_path = j(ex_path, im_name)
                im_type = im_name.split('.')[0]
                if im_type in ['t1', 't1c', 't2', 'flair']:
                    arr = im_path_to_arr(im_path).astype(np.float32)
                    brain = np.where(arr != 0)
                    mu = np.mean(arr[brain])
                    sigma = np.std(arr[brain])
                    arr[brain] = (arr[brain]-mu)/sigma
                    arr_to_im_path(arr, im_path)

# split gbm data
gbm_exs = []

for ex_name in os.listdir(gbm_in_path):
    ex_path = j(gbm_in_path, ex_name)
    if os.path.isdir(ex_path):
        gbm_exs.append((ex_path, ex_name))

shuffle(exs)

os.makedirs(gbm_out_path) 

gbm_train_path = os.path.join(gbm_out_path, 'train')
os.makedirs(gbm_train_path)

gbm_val_path = os.path.join(gbm_out_path, 'val')
os.makedirs(gbm_val_path)
for ex_path, ex_name in gbm_exs:
    os.rename(ex_path, j(gbm_val_path, ex_name))

# split lgg data
lgg_exs = []

for ex_name in os.listdir(lgg_in_path):
    ex_path = j(lgg_in_path, ex_name)
    if os.path.isdir(ex_path):
        lgg_exs.append((ex_path, ex_name))

shuffle(exs)

os.makedirs(lgg_out_path) 

lgg_train_path = os.path.join(lgg_out_path, 'train')
os.makedirs(lgg_train_path)

lgg_val_path = os.path.join(lgg_out_path, 'val')
os.makedirs(lgg_val_path)
for ex_path, ex_name in lgg_exs:
    os.rename(ex_path, j(lgg_val_path, ex_name))
