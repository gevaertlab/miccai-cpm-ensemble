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


def get_shape_v2(ex_path):
    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0].split('_')[-1]
        if im_type == 'seg':
            labels = im_path_to_arr(im_path)
    return labels.shape


def get_shape_hb(ex_path):
    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 'seg':
            labels = im_path_to_arr(im_path)
    return labels.shape


# generalize this later
def get_patch_centers(im_size):
    quotient = (im_size - 16) // 9
    remainder = (im_size - 16) % 9
    start = remainder // 2 + 12
    end = start + (quotient - 1) * 9 + 1
    return range(start, end, 9)


# hardcoded for BraTS
def get_patch_centers_fcn(im_size, patch_size, center):
    half = patch_size // 2
    im_to_patch = im_size - 2 * half
    im_rem = im_to_patch % center
    if im_rem == 0:
        start = half
        end = start + im_to_patch + center
    else:
        start = half + im_rem // 2
        end = half + im_to_patch
    return range(start, end, center)


def get_number_patches(im_size, patch_size, center_size):
    num_x = len(get_patch_centers_fcn(im_size[0], patch_size, center_size))
    num_y = len(get_patch_centers_fcn(im_size[1], patch_size, center_size))
    num_z = len(get_patch_centers_fcn(im_size[2], patch_size, center_size))
    return num_x * num_y * num_z


def normalize_image(image):
    brain = np.where(image != 0)
    mu = np.mean(image[brain])
    sigma = np.std(image[brain])
    image[brain] = (image[brain] - mu) / sigma
    return image


def preprocess_labels(labels):
    labels[labels == 4] = 3
    return labels
