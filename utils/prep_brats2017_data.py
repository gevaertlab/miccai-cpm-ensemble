import os
import gzip
import shutil

from random import shuffle


if __name__ == '__main__':
    all_data_path = 'brats2017/HGG'
    train_path = 'brats2017/train'
    val_path = 'brats2017/val'

    all_files = []

    for patient_name in os.listdir(all_data_path):
        patient_path = os.path.join(all_data_path, patient_name)
        all_files.append((patient_path, patient_name))
        for modality in os.listdir(patient_path):
            modality_path = os.path.join(patient_path, modality)
            out_path = modality_path[:-3].lower()
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
