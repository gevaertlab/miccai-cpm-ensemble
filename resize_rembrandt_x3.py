import os

import numpy as np

from utils.data_utils import im_path_to_arr
from utils.data_utils import arr_to_im_path


def resize_image(im_path, out_path):
    arr = im_path_to_arr(im_path)
    arr = np.repeat(arr, 3, axis=0)
    arr_to_im_path(arr, out_path)


if __name__ == '__main__':

    in_path_train_dir = 'data/rembrandt/train/'
    in_path_val_dir = 'data/rembrandt/val/'
    out_path_train_dir = 'data/rembrandt/resized_x3/train/'
    out_path_val_dir = 'data/rembrandt/resized_x3/val/'
    
    files_to_resize = ['flair', 't1', 't1c', 't2', 'tumor']

    if not os.path.exists(out_path_train_dir):
        os.makedirs(out_path_train_dir)
    if not os.path.exists(out_path_val_dir):
        os.makedirs(out_path_val_dir)

    print('resizing images from training set')
    for k, im_dir in enumerate(os.listdir(in_path_train_dir)):
        print('resizing image number {}, name is: {}'.format(k, im_dir))
        in_im_dir_path = os.path.join(in_path_train_dir, im_dir)
        out_im_dir_path = os.path.join(out_path_train_dir, im_dir)
        if not os.path.exists(out_im_dir_path):
            os.makedirs(out_im_dir_path)
        for im_name in os.listdir(in_im_dir_path):
           if any(x in im_name for x in files_to_resize):
                im_path = os.path.join(in_im_dir_path, im_name)
                out_path = os.path.join(out_im_dir_path, im_name)
                resize_image(im_path, out_path)

    print('resizing images from validation set')
    for k, im_dir in enumerate(os.listdir(in_path_val_dir)):
        print('resizing image number {}, name is: {}'.format(k, im_dir))
        in_im_dir_path = os.path.join(in_path_val_dir, im_dir)
        out_im_dir_path = os.path.join(out_path_val_dir, im_dir)
        if not os.path.exists(out_im_dir_path):
            os.makedirs(out_im_dir_path)
        for im_name in os.listdir(in_im_dir_path):
            if any(x in im_name for x in files_to_resize):
                im_path = os.path.join(in_im_dir_path, im_name)
                out_path = os.path.join(out_im_dir_path, im_name)
                resize_image(im_path, out_path)

