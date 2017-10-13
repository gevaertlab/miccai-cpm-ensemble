import numpy as np

import os
import shutil

import glob
import gzip

from data_utils import im_path_to_arr, arr_to_im_path

in_path = '../data/REMBRANDTVerified'
out_path = '../data/rembrandt'

def j(path, fname):
    return os.path.join(path, fname)

shutil.rmtree(j(in_path, 'HF1708=')) # this one is 512x512
shutil.rmtree(j(in_path, 'HF0899=')) # this one has no tumor

# reorganize data
for ex_name in os.listdir(in_path):
    ex_path = j(in_path, ex_name)
    if os.path.isdir(ex_path):

        with gzip.open(j(ex_path, 'tumor.nii.gz'), 'rb') as fgz:
            with open(j(ex_path, 'tumor.nii'), 'wb') as f:
                f.write(fgz.read())

        os.remove(j(ex_path, 'flair_t1_bcorr.nii'))
        os.remove(j(ex_path, 'pd_t1_bcorr.nii'))
        os.remove(j(ex_path, 'pd_t1_bcorr_brain.nii'))
        os.remove(j(ex_path, 't1-post_pre_bcorr.nii'))
        os.remove(j(ex_path, 't1-pre_bcorr.nii'))
        os.remove(j(ex_path, 't2_t1_bcorr.nii'))
        for gz in glob.glob(j(ex_path, '*.gz')):
            os.remove(gz)
        for txt in glob.glob(j(ex_path, '*.txt')):
            os.remove(txt)

        os.rename(j(ex_path, 'flair_t1_bcorr_brain.nii'),
                  j(ex_path, 'flair.nii'))
        os.rename(j(ex_path, 't1-post_pre_bcorr_brain.nii'),
                  j(ex_path, 't1.nii'))
        os.rename(j(ex_path, 't1-pre_bcorr_brain.nii'),
                  j(ex_path, 't1c.nii'))
        os.rename(j(ex_path, 't2_t1_bcorr_brain.nii'),
                  j(ex_path, 't2.nii'))

# normalize data
for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        for im_name in os.listdir(ex_path):
            im_path = os.path.join(ex_path, im_name)
            im_type = im_name.split('.')[0]
            if im_type in ['t1', 't1c', 't2', 'flair']:
                arr = im_path_to_arr(im_path).astype(np.float32)
                brain = np.where(arr != 0)
                mu = np.mean(arr[brain])
                sigma = np.std(arr[brain])
                arr[brain] = (arr[brain]-mu)/sigma
                arr_to_im_path(arr, im_path)
            if im_type == 'tumor':
                arr = im_path_to_arr(im_path).astype(np.int32)
                arr_to_im_path(arr, im_path)

# split data
test_path = os.path.join(out_path, 'test')
shutil.copytree(in_path, test_path)
