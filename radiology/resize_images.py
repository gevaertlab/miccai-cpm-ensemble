import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from eqtools.trispline import Spline

from utils.data_utils import im_path_to_arr
from utils.data_utils import arr_to_im_path

FILES_TO_RESIZE = ['flair', 't1', 't1c', 't2', 'tumor']


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


def resize_dataset(path_to_data, new_size):
    new_x, new_y, new_z = new_size
    for k, im_dir in enumerate(os.listdir(path_to_data)):
        im_dir = os.path.join(path_to_data, im_dir)
        for im_name in os.listdir(im_dir):
            if any(x in im_name for x in FILES_TO_RESIZE):
                im_path = os.path.join(im_dir, im_name)
                modality, end = im_name.split('.')
                new_name = modality + '_{}_{}_{}'.format(new_x, new_y, new_z) + '.' + end
                out_path = os.path.join(im_dir, new_name)
                resize_image(im_path, out_path, new_size)


if __name__ == '__main__':
    new_size = (100,240,240)

    path_train_dir = 'data/rembrandt/train/'
    path_val_dir = 'data/rembrandt/val/'

    print('resizing images from training set')
    resize_dataset(path_train_dir, new_size)

    print('resizing images from validation set')
    resize_dataset(path_val_dir, new_size)
