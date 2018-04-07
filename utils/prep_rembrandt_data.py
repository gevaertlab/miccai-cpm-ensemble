import os
import getpass

import shutil
import gzip

import numpy as np

from data_utils import im_path_to_arr
from data_utils import arr_to_im_path
from data_utils import get_hgg_and_lgg_patients

user_name = getpass.getuser()

in_path = '/labs/gevaertlab/data/tumor_segmentation/REMBRANDTVerified'
out_path = '/local-scratch/{}_scratch/REMBRANDTVerified'.format(user_name)
train_path = '/local-scratch/{}_scratch/rembrandt/train'.format(user_name)
val_path = '/local-scratch/{}_scratch/rembrandt/val'.format(user_name)


def j(path, fname):
    return os.path.join(path, fname)


def create_labels(directory):
    labels = [None] * 3
    if 'edema.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'edema.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'edema.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'edema.nii.gz'))
        edema = im_path_to_arr(j(directory, 'edema.nii'))
        labels[0] = edema
    if 'necrosis.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'necrosis.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'necrosis.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'necrosis.nii.gz'))
        necrosis = im_path_to_arr(j(directory, 'necrosis.nii'))
        labels[1] = necrosis
    if 'active.nii.gz' in os.listdir(directory):
        with gzip.open(j(directory, 'active.nii.gz'), 'rb') as fgz:
            with open(j(directory, 'active.nii'), 'wb') as f:
                f.write(fgz.read())
        os.remove(j(directory, 'active.nii.gz'))
        active = im_path_to_arr(j(directory, 'active.nii'))
        labels[2] = active
    return labels

print('Create a copy of the original data ...')
# copy whole dataset to new location
shutil.copytree(in_path, out_path)

# preprocess dataset
shutil.rmtree(j(out_path, 'HF1708=')) # this one is 512x512 (TODO: downsample by 2 instead?)
shutil.rmtree(j(out_path, 'HF0899=')) # this one has no tumor

print('Rename and remove data ...')
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
        if 't1-post_pre_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 't1-post_pre_bcorr.nii'))
        if 't1-pre_bcorr.nii' in os.listdir(ex_path):
            os.remove(j(ex_path, 't1-pre_bcorr.nii'))
        if 'flair2t1.txt' in os.listdir(ex_path):
            os.remove(j(ex_path, 'flair2t1.txt'))
        if 'pd2t1.txt' in os.listdir(ex_path):
            os.remove(j(ex_path, 'pd2t1.txt'))
        if 't12t1.txt' in os.listdir(ex_path):
            os.remove(j(ex_path, 't12t1.txt'))
        if 't22t1.txt' in os.listdir(ex_path):
            os.remove(j(ex_path, 't22t1.txt'))
        if 'blood.nii.gz' in os.listdir(ex_path):
            os.remove(j(ex_path, 'blood.nii.gz'))
        if '.DS_Store' in os.listdir(ex_path):
            os.remove(j(ex_path, '.DS_Store'))

        os.rename(j(ex_path, 'flair_t1_bcorr_brain.nii'),
                  j(ex_path, 'flair.nii'))
        os.rename(j(ex_path, 't1-post_pre_bcorr_brain.nii'),
                  j(ex_path, 't1c.nii'))
        os.rename(j(ex_path, 't1-pre_bcorr_brain.nii'),
                  j(ex_path, 't1.nii'))
        os.rename(j(ex_path, 't2_t1_bcorr_brain.nii'),
                  j(ex_path, 't2.nii'))


print('Create labels ...')
# create labels
for ex_name in os.listdir(out_path):
    print(ex_name)
    ex_path = os.path.join(out_path, ex_name)
    if os.path.isdir(ex_path):
        labels = create_labels(ex_path)
        curated_labels = [lab for lab in labels if lab is not None]
        shape = curated_labels[0].shape
        tumor = np.zeros(shape)
        if labels[0] is not None:
            tumor[labels[0] > 0] = 2
        if labels[1] is not None:
            tumor[labels[1] > 0] = 1
        if labels[2] is not None:
            tumor[labels[2] > 0] = 4
        arr_to_im_path(tumor, j(ex_path, 'seg.nii'))
    else:
        os.remove(ex_path)

print('Split data between HGG and LGG ...')
# split data
all_patients = os.listdir(out_path)
all_patients = [(j(out_path, pat), pat) for pat in all_patients]
all_patients = [pat for pat in all_patients if os.path.isdir(pat[0])]

#shuffle patients
np.random.seed(0)
np.random.shuffle(all_patients)

#split patients between LGG and HGG
HGG_patients_csv, LGG_patients_csv = get_hgg_and_lgg_patients(out_path)
HGG_patients_csv = [pat.decode('utf-8') for pat in HGG_patients_csv]
LGG_patients_csv = [pat.decode('utf-8') for pat in LGG_patients_csv]
HGG_patients = [pat for pat in all_patients if pat[0] in HGG_patients_csv]
LGG_patients = [pat for pat in all_patients if pat[0] in LGG_patients_csv]
other_patients = [pat for pat in all_patients if (pat[0] not in HGG_patients_csv and pat[0] not in LGG_patients_csv)]
print('number of HGG patients: %d | number of LGG patients: %d | number of other patients: %d'\
      %(len(HGG_patients), len(LGG_patients), len(other_patients)))

print('Create training and test sets ...')
#create train and test set
ratio_val = 0.3
train_patients = HGG_patients[int(ratio_val * len(HGG_patients)):]\
                 + LGG_patients[int(ratio_val * len(LGG_patients)):]\
                 + other_patients[int(ratio_val * len(other_patients)):]
val_patients = HGG_patients[:int(ratio_val * len(HGG_patients))]\
               + LGG_patients[:int(ratio_val * len(LGG_patients))]\
               + other_patients[:int(ratio_val * len(other_patients))]
np.random.shuffle(train_patients)
np.random.shuffle(val_patients)

for pat_path, pat_name in train_patients:
    copy_path = j(train_path, pat_name)
    shutil.copytree(pat_path, copy_path)

for pat_path, pat_name in val_patients:
    copy_path = j(val_path, pat_name)
    shutil.copytree(pat_path, copy_path)

shutil.rmtree(out_path)
