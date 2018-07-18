import os
import getpass

import gzip
import shutil

import numpy as np

user_name = getpass.getuser()

data_path = '/labs/gevaertlab/data/MICCAI/radiology'
out_path = '/local-scratch/{}_scratch/miccai-radiology'.format(user_name)
train_path = '/local-scratch/{}_scratch/miccai-radiology/train'.format(user_name)
val_path = '/local-scratch/{}_scratch/miccai-radiology/val'.format(user_name)

files = []

print('Unzip files ...')
for patient_name in os.listdir(data_path):
    if "html" in patient_name:
        continue
    print("Handling patient {}.".format(patient_name))
    patient_path = os.path.join(data_path, patient_name)
    out_patient_path = os.path.join(out_path, patient_name)
    files.append((out_patient_path, patient_name))
    if os.path.exists(out_patient_path):
        shutil.rmtree(out_patient_path)
    else:
        os.makedirs(out_patient_path)
    for modality in os.listdir(patient_path):
        print("\t Handling file {}.".format(modality))
        modality_path = os.path.join(patient_path, modality)
        unzipped_path = os.path.join(out_patient_path, modality[:-3])
        shutil.copy(modality_path, unzipped_path)

print('Create training and test sets ...')
# shuffle patients
np.random.seed(0)
np.random.shuffle(files)

# create train and test set
ratio_val = 0.2
train_files = files[int(ratio_val * len(files)):]
val_files = files[:int(ratio_val * len(files))]
np.random.shuffle(train_files)
np.random.shuffle(val_files)

# move data
os.makedirs(train_path)
for ex_path, ex_name in train_files:
    shutil.copytree(ex_path, os.path.join(train_path, ex_name))

os.makedirs(val_path)
for ex_path, ex_name in val_files:
    shutil.copytree(ex_path, os.path.join(val_path, ex_name))

shutil.rmtree(out_path)
