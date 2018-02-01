import os
import shutil

import glob
import gzip

import numpy as np

from data_utils import im_path_to_arr
from data_utils import arr_to_im_path


in_path = '/labs/gevaertlab/data/tumor_segmentation/rembrandt'
train_path = '/local-scratch/romain_scratch/rembrandt/train'
val_path = '/local-scratch/romain_scratch/rembrandt/val'


def j(path, fname):
    return os.path.join(path, fname)


def create_labels(directory):
    # hardcoded for Rembrandt dataset
    labels = np.zeros((55, 256, 256))
    edema = im_path_to_arr(j(directory, 'edema.nii'))
    necrosis = im_path_to_arr(j(directory, 'necrosis.nii'))
    active = im_path_to_arr(j(directory, 'active.nii'))
    labels[edema > 0] = 2
    labels[active > 0] = 4
    labels[necrosis > 0] = 1
    arr_to_im_path(labels, j(directory, 'tumor.nii'))


shutil.rmtree(j(in_path, 'HF1708=')) # this one is 512x512
shutil.rmtree(j(in_path, 'HF0899=')) # this one has no tumor


# reorganize data
for ex_name in os.listdir(in_path):
    ex_path = j(in_path, ex_name)
    if os.path.isdir(ex_path):

        with gzip.open(j(ex_path, 'edema.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'edema.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'active.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'active.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'necrosis.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'necrosis.nii'), 'wb') as f:
                f.write(fgz.read())

        os.remove(j(ex_path, 'flair_t1_bcorr.nii'))
        os.remove(j(ex_path, 'pd_t1_bcorr.nii'))
        os.remove(j(ex_path, 'pd_t1_bcorr_brain.nii'))
        os.remove(j(ex_path, 't1-post_pre_bcorr.nii'))
        os.remove(j(ex_path, 't1-pre_bcorr.nii'))
        os.remove(j(ex_path, 't2_t1_bcorr.nii'))

        os.rename(j(ex_path, 'flair_t1_bcorr_brain.nii'),
                  j(ex_path, 'flair.nii'))
        os.rename(j(ex_path, 't1-post_pre_bcorr_brain.nii'),
                  j(ex_path, 't1.nii'))
        os.rename(j(ex_path, 't1-pre_bcorr_brain.nii'),
                  j(ex_path, 't1c.nii'))
        os.rename(j(ex_path, 't2_t1_bcorr_brain.nii'),
                  j(ex_path, 't2.nii'))


# create images and labels
for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        for im_name in os.listdir(ex_path):
            im_path = os.path.join(ex_path, im_name)
            im_type = im_name.split('.')[0]
            if im_type in ['t1', 't1c', 't2', 'flair']:
                arr = im_path_to_arr(im_path).astype(np.float32)
                arr_to_im_path(arr, im_path)
        create_labels(ex_path)


# split data
all_patients = os.listdir(in_path)
all_patients = [(j(in_path, pat), pat) for pat in all_patients]
all_patients = [pat for pat in all_patients if os.path.isdir(pat[0])]
np.random.shuffle(all_patients)

train_patients = all_patients[:int(0.8 * len(all_patients))]
val_patients = all_patients[int(0.8 * len(all_patients)):]

for pat_path, pat_name in train_patients:
    copy_path = j(train_path, pat_name)
    shutil.copytree(pat_path, copy_path)

for pat_path, pat_name in val_patients:
    copy_path = j(val_path, pat_name)
    shutil.copytree(pat_path, copy_path)
