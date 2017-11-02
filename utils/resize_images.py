import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from eqtools.trispline import Spline

from utils.data_utils import im_path_to_arr
from utils.data_utils import arr_to_im_path


def resize_image(im_path, out_path, new_size):
    arr = im_path_to_arr(im_path)
    nx, ny, nz = arr.shape
    xx = np.linspace(1, nx, nx)
    yy = np.linspace(1, ny, ny)
    zz = np.linspace(1, nz, nz)

    interpolating_function_tricubic = Spline(xx, yy, zz, arr)
    # interpolating_function = RegularGridInterpolator((xx, yy, zz), arr, method="linear")
    # interpolating_function_nearest = RegularGridInterpolator((xx, yy, zz), arr, method="nearest")

    new_x, new_y, new_z = new_size
    xi = np.linspace(1, nx, new_x)
    yi = np.linspace(1, ny, new_y)
    zi = np.linspace(1, nz, new_z)

    new_image = np.zeros((new_x, new_y, new_z))
    for i in range(new_x):
        for j in range(new_y):
            for k in range(new_z):
                new_image[i][j][k] = interpolating_function_tricubic.ev(xi[i], yi[j], zi[k])

    arr_to_im_path(new_image, out_path)


if __name__ == '__main__':
    new_size = (100,240,240)

    in_path_train_dir = 'data/train/'
    in_path_val_dir = 'data/val/'
    out_path_train_dir = 'data/resized_{}/train/'.format(str(new_size))
    out_path_val_dir = 'data/resized_{}/val/'.format(str(new_size))

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
            im_path = os.path.join(in_im_dir_path, im_name)
            out_path = os.path.join(out_im_dir_path, im_name)
            resize_image(im_path, out_path, new_size)

    print('resizing images from validation set')
    for k, im_dir in enumerate(os.listdir(in_path_val_dir)):
        print('resizing image number {}, name is: {}'.format(k, im_dir))
        in_im_dir_path = os.path.join(in_path_val_dir, im_dir)
        out_im_dir_path = os.path.join(out_path_val_dir, im_dir)
        if not os.path.exists(out_im_dir_path):
            os.makedirs(out_im_dir_path)
        for im_name in os.listdir(in_im_dir_path):
            im_path = os.path.join(in_im_dir_path, im_name)
            out_path = os.path.join(out_im_dir_path, im_name)
            resize_image(im_path, out_path, new_size)

    