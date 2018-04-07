import os
import getpass

import gzip
import shutil

import numpy as np

user_name = getpass.getuser()

HGG_data_path = '/labs/gevaertlab/data/tumor_segmentation/brats2017/HGG'
LGG_data_path = '/labs/gevaertlab/data/tumor_segmentation/brats2017/LGG'
out_path = '/local-scratch/{}_scratch/brats2017full'.format(user_name)
train_path = '/local-scratch/{}_scratch/brats2017/train'.format(user_name)
val_path = '/local-scratch/{}_scratch/brats2017/val'.format(user_name)

HGG_files = []
LGG_files = []

print('Unzip files ...')
for patient_name in os.listdir(HGG_data_path):
    patient_path = os.path.join(HGG_data_path, patient_name)
    out_patient_path = os.path.join(out_path, patient_name)
    HGG_files.append((out_patient_path, patient_name))
    if os.path.exists(out_patient_path):
        shutil.rmtree(out_patient_path)
    else:
        os.makedirs(out_patient_path)
    for modality in os.listdir(patient_path):
        modality_path = os.path.join(patient_path, modality)
        unzipped_path = os.path.join(out_patient_path, modality[:-3])
        with gzip.open(modality_path, 'rb') as fgz:
            with open(unzipped_path, 'wb') as f:
                f.write(fgz.read())

for patient_name in os.listdir(LGG_data_path):
    patient_path = os.path.join(LGG_data_path, patient_name)
    out_patient_path = os.path.join(out_path, patient_name)
    LGG_files.append((out_patient_path, patient_name))
    if os.path.exists(out_patient_path):
        shutil.rmtree(out_patient_path)
    else:
        os.makedirs(out_patient_path)
    for modality in os.listdir(patient_path):
        modality_path = os.path.join(patient_path, modality)
        unzipped_path = os.path.join(out_patient_path, modality[:-3])
        with gzip.open(modality_path, 'rb') as fgz:
            with open(unzipped_path, 'wb') as f:
                f.write(fgz.read())

print('Create training and test sets ...')
#shuffle patients
np.random.seed(0)
np.random.shuffle(HGG_files)
np.random.shuffle(LGG_files)

#create train and test set
ratio_val = 0.2
train_files = HGG_files[int(ratio_val * len(HGG_files)):] + LGG_files[int(ratio_val * len(LGG_files)):]
val_files = HGG_files[:int(ratio_val * len(HGG_files))] + LGG_files[:int(ratio_val * len(LGG_files))]
np.random.shuffle(train_files)
np.random.shuffle(val_files)

#move data
os.makedirs(train_path)
for ex_path, ex_name in train_files:
    shutil.copytree(ex_path, os.path.join(train_path, ex_name))

os.makedirs(val_path)
for ex_path, ex_name in val_files:
    shutil.copytree(ex_path, os.path.join(val_path, ex_name))

shutil.rmtree(out_path)
