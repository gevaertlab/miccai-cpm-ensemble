import os

import numpy as np
import tensorflow as tf

from utils.data_utils import im_path_to_arr
from utils.data_utils import normalize_image
from utils.data_utils import preprocess_labels 


def load_data_brats(patient_path):
    data = [None] * 4

    patient_path = patient_path.decode('utf-8')

    for im_name in os.listdir(patient_path):
        im_path = os.path.join(patient_path, im_name)
        # for Brats2017
        im_type = im_name.split('.')[0].split('_')[-1]
        image = im_path_to_arr(im_path)
        if im_type == 't1':
            data[0] = normalize_image(image)
        if im_type == 't1c' or im_type == 't1ce':
            data[1] = normalize_image(image)
        if im_type == 't2':
            data[2] = normalize_image(image)
        if im_type == 'flair' or im_type == 'fla':
            data[3] = normalize_image(image)
        if im_type == 'tumor' or im_type == 'seg':
            labels = preprocess_labels(image)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)
    return data, labels


def train_data_iter(patient_path, batch_size, patch_size):
    data, labels = load_data_brats(patient_path)

    half_patch = patch_size // 2

    trimmed_data = data[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch, :]
    trimmed_labels = labels[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch]

    bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
    fg = np.where((trimmed_labels > 0) & (trimmed_data[..., 0] != 0))

    num_bg = len(bg[0])
    num_fg = len(fg[0])

    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        if np.random.rand() <= 0.5:
            idx = np.random.randint(num_bg)
            i = bg[0][idx]
            j = bg[1][idx]
            k = bg[2][idx]
            x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
            y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]
        else:
            idx = np.random.randint(num_fg)
            i = fg[0][idx]
            j = fg[1][idx]
            k = fg[2][idx]
            x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
            y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]

        i_batch.append(i)
        j_batch.append(j)
        k_batch.append(k)
        x_batch.append(x)
        y_batch.append(y)

    path_batch = np.array([patient_path] * batch_size)
    i_batch = np.array(i_batch).astype(np.int32)
    j_batch = np.array(j_batch).astype(np.int32)
    k_batch = np.array(k_batch).astype(np.int32)
    x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch]).astype(np.float32)
    y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch]).astype(np.int32)

    # because we use the same iterator for both train and test, we need to have the same outputs
    # we create i_batch, j_batch and k_batch for training, even if we don't use it
    return path_batch, i_batch, j_batch, k_batch, x_batch, y_batch


def test_data_iter(patient_path, data_path, patch_size):
    data, labels = load_data_brats(patient_path)

    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []

    i_len, j_len, k_len = labels.shape

    i_rem = i_len % patch_size
    j_rem = j_len % patch_size
    k_rem = k_len % patch_size

    for i in range(i_rem // 2, i_len, patch_size):
        for j in range(j_rem // 2, j_len, patch_size):
            for k in range(k_rem // 2, k_len, patch_size):

                if (i + patch_size) <= i_len and (j + patch_size) <= j_len and (k + patch_size) <= k_len:

                    x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                    y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]

                    i_batch.append(i)
                    j_batch.append(j)
                    k_batch.append(k)
                    x_batch.append(x)
                    y_batch.append(y)

    path_batch = np.array([patient_path] * len(i_batch))
    i_batch = np.array(i_batch).astype(np.int32)
    j_batch = np.array(j_batch).astype(np.int32)
    k_batch = np.array(k_batch).astype(np.int32)
    x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch]).astype(np.float32)
    y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch]).astype(np.int32)
    
    return path_batch, i_batch, j_batch, k_batch, x_batch, y_batch


def get_dataset(directory, is_test, batch_size, patch_size, num_workers=4):
    patients = os.listdir(directory)
    patients = [os.path.join(directory, pat) for pat in patients]
    # need to encode in bytes to pass it to tf.py_func
    patient = [pat.encode('utf-8') for pat in patients]
    patients = tf.constant(patients)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((patients, patients, patients, patients, patients, patients))

    if not is_test:
        dataset = dataset.map(lambda p, i, j, k, x, y: tuple(tf.py_func(train_data_iter,
                                                               [x, batch_size, patch_size],
                                                               [tf.string, tf.int32, tf.int32,\
                                                                tf.int32, tf.float32, tf.int32])),
                              num_threads=num_workers,
                              output_buffer_size=batch_size)
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(buffer_size=5000)
    else:
        dataset = dataset.map(lambda p, i, j, k, x, y: tuple(tf.py_func(test_data_iter,
                                                               [x, batch_size, patch_size],
                                                               [tf.string, tf.int32, tf.int32,\
                                                                tf.int32, tf.float32, tf.int32])),
                              num_threads=num_workers,
                              output_buffer_size=batch_size)
        dataset = dataset.unbatch()
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset
