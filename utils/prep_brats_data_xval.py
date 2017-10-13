import numpy as np

import os
import shutil

from random import shuffle

from data_utils import im_path_to_arr, arr_to_im_path

in_path = '/share/PI/ogevaert/shirley/brats_hgg/brats16' # '/share/PI/ogevaert/brats_hgg'
out_base_path = '/share/PI/ogevaert/brats_hgg_xval'

n_val = 55

exs = []

train_path = os.path.join(in_path, 'train')
for ex_name in os.listdir(train_path):
    ex_path = os.path.join(train_path, ex_name)
    if os.path.isdir(ex_path):
        exs.append((ex_path, ex_name))
        
val_path = os.path.join(in_path, 'val')
for ex_name in os.listdir(val_path):
    ex_path = os.path.join(val_path, ex_name)
    if os.path.isdir(ex_path):
        exs.append((ex_path, ex_name))

n_split = len(exs)/n_val

for split in range(n_split):
    
    out_path = out_base_path + '_' + str(split)

    train_exs = exs[:split*n_val] + exs[(split+1)*n_val:]
    val_exs = exs[split*n_val:(split+1)*n_val]

    os.makedirs(out_path) 

    train_path = os.path.join(out_path, 'train')
    os.makedirs(train_path)
    for ex_path, ex_name in train_exs:
        os.rename(ex_path, os.path.join(train_path, ex_name))

    val_path = os.path.join(out_path, 'val')
    os.makedirs(val_path)
    for ex_path, ex_name in val_exs:
        os.rename(ex_path, os.path.join(val_path, ex_name))
