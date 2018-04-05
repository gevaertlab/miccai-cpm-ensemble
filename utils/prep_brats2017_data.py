import os
import getpass

import gzip
import shutil

from random import shuffle

user_name = getpass.getuser()

if __name__ == '__main__':
    HGG_data_path = '/labs/gevaertlab/data/tumor_segmentation/brats2017/HGG'
    LGG_data_path = '/labs/gevaertlab/data/tumor_segmentation/brats2017/LGG'
    train_path = '/local-scratch/{}_scratch/brats2017/train'.format(user_name)
    val_path = '/local-scratch/{}_scratch/brats2017/val'.format(user_name)

    all_files = []

    for patient_name in os.listdir(HGG_data_path):
        patient_path = os.path.join(HGG_data_path, patient_name)
        all_files.append((patient_path, patient_name))
        for modality in os.listdir(patient_path):
            modality_path = os.path.join(patient_path, modality)
            out_path = modality_path[:-3]
            with gzip.open(modality_path, 'rb') as fgz:
                with open(out_path, 'wb') as f:
                    f.write(fgz.read())
            os.remove(modality_path)

    for patient_name in os.listdir(LGG_data_path):
        patient_path = os.path.join(LGG_data_path, patient_name)
        all_files.append((patient_path, patient_name))
        for modality in os.listdir(patient_path):
            modality_path = os.path.join(patient_path, modality)
            out_path = modality_path[:-3]
            with gzip.open(modality_path, 'rb') as fgz:
                with open(out_path, 'wb') as f:
                    f.write(fgz.read())
            os.remove(modality_path)

    shuffle(all_files)

    nb_val = int(0.2 * len(all_files))
    train_files = all_files[nb_val:]
    val_files = all_files[:nb_val]

    os.makedirs(train_path)
    for ex_path, ex_name in train_files:
        shutil.copytree(ex_path, os.path.join(train_path, ex_name))

    os.makedirs(val_path)
    for ex_path, ex_name in val_files:
        shutil.copytree(ex_path, os.path.join(val_path, ex_name))
