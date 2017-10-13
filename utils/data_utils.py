import os

import numpy as np
import SimpleITK as sitk

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

def get_shape(ex_path):
    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 'tumor':
            labels = im_path_to_arr(im_path)
    return labels.shape

def get_shape_hb(ex_path):
    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 'seg':
            labels = im_path_to_arr(im_path)
    return labels.shape
