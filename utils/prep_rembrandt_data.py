import os
import shutil

import glob
import gzip

import numpy as np

from data_utils import im_path_to_arr
from data_utils import arr_to_im_path


in_path = '/labs/gevaertlab/data/tumor_segmentation/REMBRANDTVerified'
out_path = '/local-scratch/romain_scratch/REMBRANDTVerified'
train_path = '/local-scratch/romain_scratch/rembrandt/train'
val_path = '/local-scratch/romain_scratch/rembrandt/val'


def j(path, fname):
    return os.path.join(path, fname)


def create_labels(directory):
    # hardcoded for Rembrandt dataset
    labels = np.zeros((55, 256, 256))
    if 'edema.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'edema.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'edema.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'edema.nii'))
        edema = im_path_to_arr(j(directory, 'edema.nii'))
        labels[edema > 0] = 2
    if 'necrosis.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'necrosis.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'necrosis.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'necrosis.nii'))
        necrosis = im_path_to_arr(j(directory, 'necrosis.nii'))
        labels[necrosis > 0] = 1
    if 'active.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'active.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'active.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'active.nii'))
        active = im_path_to_arr(j(directory, 'active.nii'))
        labels[active > 0] = 4
    arr_to_im_path(labels, j(directory, 'tumor.nii'))

# copy whole dataset to new location
shutil.copytree(in_path, out_path)

# preprocess dataset
shutil.rmtree(j(out_path, 'HF1708=')) # this one is 512x512
shutil.rmtree(j(out_path, 'HF0899=')) # this one has no tumor


# reorganize data
for ex_name in os.listdir(out_path):
    ex_path = j(out_path, ex_name)
    if os.path.isdir(ex_path):

        if 'flair_t1_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 'flair_t1_bcorr.nii'))
        if 'pd_t1_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 'pd_t1_bcorr.nii'))
        if 'pd_t1_bcorr_brain.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 'pd_t1_bcorr_brain.nii'))
        if 'post_pre_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 't1-post_pre_bcorr.nii'))
        if 'pre_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 't1-pre_bcorr.nii'))
        if 't2_t1_bcorr.nii' in os.listdir(ex_path):
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
for ex_name in os.listdir(out_path):
    print(ex_name)
    ex_path = os.path.join(out_path, ex_name)
    if os.path.isdir(ex_path):
        for im_name in os.listdir(ex_path):
            im_path = os.path.join(ex_path, im_name)
            im_type = im_name.split('.')[0]
            if im_type in ['t1', 't1c', 't2', 'flair']:
                arr = im_path_to_arr(im_path).astype(np.float32)
                arr_to_im_path(arr, im_path)
        create_labels(ex_path)


# split data
all_patients = os.listdir(out_path)
all_patients = [(j(out_path, pat), pat) for pat in all_patients]
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

shutil.rmtree(out_path)
