import os

import numpy as np
import tensorflow as tf

from utils.data_utils import im_path_to_arr
from utils.data_utils import normalize_image
from utils.data_utils import preprocess_labels 
from utils.data_utils import get_patch_centers_fcn 


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
    try:
        data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)
    except:
        print('data cannot be concat for patient:', patient_path)
        assert(False)
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
            x = data[i - half_patch:i + half_patch, j - half_patch:j + half_patch, k - half_patch:k + half_patch, :]
            y = labels[i - half_patch:i + half_patch, j - half_patch:j + half_patch, k - half_patch:k + half_patch]
        else:
            idx = np.random.randint(num_fg)
            i = fg[0][idx]
            j = fg[1][idx]
            k = fg[2][idx]
            x = data[i - half_patch:i + half_patch, j - half_patch:j + half_patch, k - half_patch:k + half_patch, :]
            y = labels[i - half_patch:i + half_patch, j - half_patch:j + half_patch, k - half_patch:k + half_patch]

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


def train_data_iter_v2(patient_path, batch_size, patch_size):
    data, labels = load_data_brats(patient_path)

    half_patch = patch_size // 2

    trimmed_data = data[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch, :]
    trimmed_labels = labels[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch]

    bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
    enhanced = np.where((trimmed_labels == 1) & (trimmed_data[..., 0] != 0))
    necrotic = np.where((trimmed_labels == 3) & (trimmed_data[..., 0] != 0))
    edema = np.where((trimmed_labels == 2) & (trimmed_data[..., 0] != 0))
    fg = np.where((trimmed_labels > 0) & (trimmed_data[..., 0] != 0))

    num_bg = len(bg[0])
    num_enhanced = len(enhanced[0])
    num_necrotic = len(necrotic[0])
    num_edema = len(edema[0])
    num_fg = len(fg[0])
    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []

    """
    oversampling:
        - 50% non tumorous
        - 20% enhanced
        - 10% necoritc
        - 20% edema
    """
    ratio_non_tumorous = 0.5
    ratio_enhanced = 0.7
    ratio_necrotic = 0.8
    ratio_edema = 1.0
    for _ in range(batch_size):
        epsilon = np.random.rand()
        if epsilon <= ratio_non_tumorous:
            idx = np.random.randint(num_bg)
            i = bg[0][idx]
            j = bg[1][idx]
            k = bg[2][idx]
        elif epsilon <= ratio_enhanced:
            idx = np.random.randint(num_enhanced)
            i = enhanced[0][idx]
            j = enhanced[1][idx]
            k = enhanced[2][idx]
        elif epsilon <= ratio_necrotic and num_necrotic > 0:
            idx = np.random.randint(num_necrotic)
            i = necrotic[0][idx]
            j = necrotic[1][idx]
            k = necrotic[2][idx]
        else:
            idx = np.random.randint(num_edema)
            i = edema[0][idx]
            j = edema[1][idx]
            k = edema[2][idx]
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


def test_data_iter(patient_path, patch_size):
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


def test_data_iter_v2(all_patients, patch_size, center_size, batch_size):
    batch_count = 0

    half_patch = patch_size // 2
    half_center = center_size // 2
    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []
    path_batch = []

    for patient_path in all_patients:

        data, labels = load_data_brats(patient_path)
        i_len, j_len, k_len = labels.shape

        for i in get_patch_centers_fcn(i_len, patch_size, center_size):
            for j in get_patch_centers_fcn(j_len, patch_size, center_size):
                for k in get_patch_centers_fcn(k_len, patch_size, center_size):

                    x = data[i - half_patch:i + half_patch,\
                             j - half_patch:j + half_patch,\
                             k - half_patch:k + half_patch, :]
                    y = labels[i - half_center:i + half_center,\
                               j - half_center:j + half_center,\
                               k - half_center:k + half_center]

                    i_batch.append(i)
                    j_batch.append(j)
                    k_batch.append(k)
                    x_batch.append(x)
                    y_batch.append(y)
                    path_batch.append(patient_path)

                    batch_count += 1
                    if batch_count == batch_size:
                        path_batch = np.array(path_batch)
                        i_batch = np.array(i_batch).astype(np.int32)
                        j_batch = np.array(j_batch).astype(np.int32)
                        k_batch = np.array(k_batch).astype(np.int32)
                        x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch]).astype(np.float32)
                        y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch]).astype(np.int32)
                        
                        yield path_batch, i_batch, j_batch, k_batch, x_batch, y_batch

                        i_batch = []
                        j_batch = []
                        k_batch = []
                        x_batch = []
                        y_batch = []
                        path_batch = []

                        batch_count = 0

    if batch_count != 0:  
        path_batch = np.array(path_batch)
        i_batch = np.array(i_batch).astype(np.int32)
        j_batch = np.array(j_batch).astype(np.int32)
        k_batch = np.array(k_batch).astype(np.int32)
        x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch]).astype(np.float32)
        y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch]).astype(np.int32)
        
        yield path_batch, i_batch, j_batch, k_batch, x_batch, y_batch


def get_dataset(directory, is_test, batch_size, patch_size, num_workers=4):
    patients = os.listdir(directory)
    patients = [os.path.join(directory, pat) for pat in patients]
    # need to encode in bytes to pass it to tf.py_func
    patients = [pat.encode('utf-8') for pat in patients]

    if not is_test:
        patients = tf.constant(patients)
        dataset = tf.data.Dataset.from_tensor_slices((patients, patients, patients, patients, patients, patients))
        dataset = dataset.map(lambda p, i, j, k, x, y: tuple(tf.py_func(train_data_iter_v2,
                                                               [x, batch_size, patch_size],
                                                               [tf.string, tf.int32, tf.int32,\
                                                                tf.int32, tf.float32, tf.int32])),
                              num_parallel_calls=num_workers)
        dataset = dataset.apply(tf.contrib.data.unbatch())
        dataset = dataset.shuffle(buffer_size=5000)
    else:
        def gen():
            return test_data_iter_v2(patients, patch_size, 10, batch_size)

        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.string, tf.int32, tf.int32,\
                                                               tf.int32, tf.float32, tf.int32))
        # dataset = dataset.map(lambda p, i, j, k, x, y: tuple(tf.py_func(test_data_iter,
        #                                                        [x, patch_size],
        #                                                        [tf.string, tf.int32, tf.int32,\
        #                                                         tf.int32, tf.float32, tf.int32])),
        #                       num_threads=num_workers,
        #                       output_buffer_size=batch_size)
        dataset = dataset.apply(tf.contrib.data.unbatch())
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset


def get_dataset_single_patient(patient, batch_size, patch_size):
    def gen():
        return test_data_iter_v2([patient], patch_size, 10, batch_size)
    dataset = tf.data.Dataset.from_generator(generator=gen,
                                             output_types=(tf.string, tf.int32, tf.int32,\
                                                           tf.int32, tf.float32, tf.int32))
    dataset = dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.batch(batch_size)
    return dataset
