import numpy as np

import os
import shutil

from random import shuffle

from data_utils import im_path_to_arr, arr_to_im_path

in_path = '../data/BRATS2015_Training/HGG'
out_path = '../data/brats_hgg_unnorm'

# reorganize data
for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        for mod_name in os.listdir(ex_path):
            mod_path = os.path.join(ex_path, mod_name)
            if os.path.isdir(mod_path):
                modality = mod_name.split('.')[4]
                im_name = mod_name + '.mha'
                im_path = os.path.join(mod_path, im_name)
                if modality == 'MR_T1':
                    os.rename(im_path, os.path.join(ex_path, 't1.mha'))
                if modality == 'MR_T1c':
                    os.rename(im_path, os.path.join(ex_path, 't1c.mha'))
                if modality == 'MR_T2':
                    os.rename(im_path, os.path.join(ex_path, 't2.mha'))
                if modality == 'MR_Flair':
                    os.rename(im_path, os.path.join(ex_path, 'flair.mha'))
                if modality == 'OT':
                    os.rename(im_path, os.path.join(ex_path, 'tumor.mha'))
                shutil.rmtree(mod_path)

# normalize data
for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        for im_name in os.listdir(ex_path):
            im_path = os.path.join(ex_path, im_name)
            im_type = im_name.split('.')[0]
            if im_type == 'tumor':
                arr = (im_path_to_arr(im_path) > 0).astype(np.int32)
                arr_to_im_path(arr, im_path)

# move data
exs = []

for ex_name in os.listdir(in_path):
    ex_path = os.path.join(in_path, ex_name)
    if os.path.isdir(ex_path):
        exs.append((ex_path, ex_name))

shuffle(exs)

os.makedirs(out_path) 
for ex_path, ex_name in exs:
    os.rename(ex_path, os.path.join(out_path, ex_name))
