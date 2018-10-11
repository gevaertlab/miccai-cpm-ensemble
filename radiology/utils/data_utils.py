import os

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize


def im_path_to_arr(im_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(im_path))


def arr_to_im_path(arr, im_path):
    sitk.WriteImage(sitk.GetImageFromArray(arr), im_path)


def get_ex_paths(path):
    ex_paths = []
    for ex_name in os.listdir(path):
        ex_path = os.path.join(path, ex_name)
        if os.path.isdir(ex_path):
            ex_paths.append(ex_path)
    return ex_paths


def normalize_image(image):
    brain = np.where(image != 0)
    mu = np.mean(image[brain])
    sigma = np.std(image[brain])
    image[brain] = (image[brain] - mu) / sigma
    return image


def resize_raw_to_base(data):
    data[np.isnan(data)] = 0.
    M = np.max(data)
    m = np.min(data)
    data = (data - m) / (M - m)
    data = np.rollaxis(data, 1, 0)
    data = np.rollaxis(data, 2, 1)
    return resize(data, (320, 320, 24))
