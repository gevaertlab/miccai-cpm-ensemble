import os

import numpy as np
import pandas as pd
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
        if 'seg.nii' in im_name:
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
    # in BraTS and rembrandt the class 'enhancing' has label 4 and there is no label 3
    # need to do that for the cross entropy loss
    labels[labels == 4] = 3
    return labels


def remove_low_high(image):
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)
    image[image < low] = low
    image[image > high] = high
    return image


def resize_data_to_brats_size(data):
    # hardcoded for brats 2017
    shape_of_brats = (155, 240, 240)
    current_shape = data.shape
    ratio_x = shape_of_brats[0] / current_shape[0]
    ratio_y = shape_of_brats[1] / current_shape[1]
    ratio_z = shape_of_brats[2] / current_shape[2]

    if ratio_x > 1 and round(ratio_x) >= 2:
        data = np.repeat(data, round(ratio_x), axis=0)
    if ratio_x <= 0.5:
        down = round(1 / ratio_x)
        data = data[::down, ...]

    if ratio_y > 1 and round(ratio_y) >= 2:
        data = np.repeat(data, round(ratio_y), axis=1)
    if ratio_y <= 0.5:
        down = round(1 / ratio_y)
        data = data[:, ::down, ...]

    if ratio_z > 1 and round(ratio_z) >= 2:
        data = np.repeat(data, round(ratio_z), axis=2)
    if ratio_z <= 0.5:
        down = round(1 / ratio_z)
        data = data[:, :, ::down, ...]

    return data

def resize_data_to_original_size(data, original_shape):
    current_shape = data.shape
    ratio_x = current_shape[0] / original_shape[0]
    ratio_y = current_shape[1] / original_shape[1]
    ratio_z = current_shape[2] / original_shape[2]
    
    if ratio_x > 1 and round(ratio_x) >= 2:
        data = data[::round(ratio_x), :, :]
    if ratio_x <= 0.5:
        down = round(1 / ratio_x)
        data = np.repeat(data, down, axis=0)
       
    if ratio_y > 1 and round(ratio_y) >= 2:
        data = data[:, ::round(ratio_y), :]
    if ratio_y <= 0.5:
        down = round(1 / ratio_y)
        data = np.repeat(data, down, axis=1)
        
    if ratio_z > 1 and round(ratio_z) >= 2:
        data = data[:, :, ::round(ratio_z)]
    if ratio_z <= 0.5:
        down = round(1 / ratio_z)
        data = np.repeat(data, down, axis=2)

    return data


def get_hgg_and_lgg_patients(val_path):
    HGG_patients = []
    if 'brats' in val_path.lower():
        HGG_patients = os.listdir('/labs/gevaertlab/data/tumor_segmentation/brats2017/HGG')
        LGG_patients = os.listdir('/labs/gevaertlab/data/tumor_segmentation/brats2017/LGG')
    if 'rembrandt' in val_path.lower():
        df = pd.read_csv('/labs/gevaertlab/data/tumor_segmentation/REMBRANDT_Clinical_Annotation_Updated.csv')
        df = df.loc[:, ['SAMPLE_ID', 'DISEASE_TYPE']]
        df_hgg = df[df.DISEASE_TYPE == 'GBM']
        HGG_patients = df_hgg.SAMPLE_ID.tolist()
        HGG_patients = [pat + '=' for pat in HGG_patients]
        df_lgg = df[df.DISEASE_TYPE != 'GBM']
        LGG_patients = df_lgg.SAMPLE_ID.tolist()
        LGG_patients = [pat + '=' for pat in LGG_patients]
    
    HGG_patients = [os.path.join(val_path, pat) for pat in HGG_patients]
    HGG_patients = [pat.encode('utf-8') for pat in HGG_patients]
    LGG_patients = [os.path.join(val_path, pat) for pat in LGG_patients]
    LGG_patients = [pat.encode('utf-8') for pat in LGG_patients]
    return HGG_patients, LGG_patients
