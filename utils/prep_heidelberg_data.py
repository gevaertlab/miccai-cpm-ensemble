from __future__ import print_function

import numpy as np

import os

from random import shuffle

import glob
import gzip

from data_utils import im_path_to_arr, arr_to_im_path

in_path = '/share/PI/ogevaert/sample'
out_path = '/share/PI/ogevaert/heidelberg'

n_val = 3

def j(path, fname):
    return os.path.join(path, fname)

# reorganize data
print('reorganize')

for ex_name in os.listdir(in_path):
    ex_path = j(in_path, ex_name)
    if os.path.isdir(ex_path):

        with gzip.open(j(ex_path, 'T1_mutualinfo2_bet.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 't1.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'T1KM_mutualinfo2_reg.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 't1c.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'T1KM_mutualinfo2_sub.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 't1c_sub.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'T2_mutualinfo2_reg.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 't2.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'FLAIR_mutualinfo2_reg.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'flair.nii'), 'wb') as f:
                f.write(fgz.read())

        with gzip.open(j(ex_path, 'seg_classes_bet_bin.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'seg.nii'), 'wb') as f:
                f.write(fgz.read())

        for gz in glob.glob(j(ex_path, '*.gz')):
            os.remove(gz)

# normalize data
print('normalize')

for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        for im_name in os.listdir(ex_path):
            im_path = os.path.join(ex_path, im_name)
            im_type = im_name.split('.')[0]
            if im_type in ['t1', 't1c', 't1c_sub', 't2', 'flair']:
                arr = im_path_to_arr(im_path).astype(np.float32)
                brain = np.where(arr != 0)
                mu = np.mean(arr[brain])
                sigma = np.std(arr[brain])
                arr[brain] = (arr[brain]-mu)/sigma
                arr_to_im_path(arr, im_path)
            if im_type == 'seg':
                arr = im_path_to_arr(im_path).astype(np.int32)
                arr_to_im_path(arr, im_path)

# split data
print('split')

exs = []

for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        exs.append((ex_path, ex_name))

shuffle(exs)

train_exs = exs[n_val:]
val_exs = exs[:n_val]

os.makedirs(out_path) 

train_path = os.path.join(out_path, 'train')
os.makedirs(train_path)
for ex_path, ex_name in train_exs:
    os.rename(ex_path, os.path.join(train_path, ex_name))

val_path = os.path.join(out_path, 'val')
os.makedirs(val_path)
for ex_path, ex_name in val_exs:
    os.rename(ex_path, os.path.join(val_path, ex_name))

os.rmdir(in_path)
