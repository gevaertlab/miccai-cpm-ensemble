import numpy as np

import os
import shutil
import random

from random import shuffle

from data_utils import im_path_to_arr, arr_to_im_path

in_path = '/share/PI/ogevaert/shirley/brats_hgg/brats16' # '/share/PI/ogevaert/brats_hgg'
out_path = '/share/PI/ogevaert/shirley/brats_hgg'

val_frac = 0.2

exs = []

train_path = in_path
for ex_name in os.listdir(train_path):
    ex_path = os.path.join(train_path, ex_name)
    if os.path.isdir(ex_path):
        exs.append((ex_path, ex_name))

print(len(exs))
# print(exs)
val_exs = random.sample(exs, int(val_frac * len(exs)))
print(len(val_exs))



val_path = os.path.join(out_path, 'val')
os.makedirs(val_path)

for ex_path, ex_name in val_exs:
    os.rename(ex_path, os.path.join(val_path, ex_name))
