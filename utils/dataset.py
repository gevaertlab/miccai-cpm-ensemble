import os

import numpy as np
import tensorflow as tf

from utils.data_utils import im_path_to_arr
from utils.data_utils import normalize_image
from utils.data_utils import preprocess_labels
from utils.data_utils import get_patch_centers_fcn


def load_data_brats(patient_path, is_test, modalities):
    data = [None] * 4
    patient_path = patient_path.decode('utf-8')

    for im_name in os.listdir(patient_path):
        im_path = os.path.join(patient_path, im_name)
        # for Brats2017
        im_type = im_name.split('.')[0].split('_')[-1]
        image = im_path_to_arr(im_path)
        if im_type == 't1' and modalities[0]:
            image = normalize_image(image)
            data[0] = image
        if (im_type == 't1c' or im_type == 't1ce') and modalities[1]:
            image = normalize_image(image)
            data[1] = image
        if im_type == 't2' and modalities[2]:
            image = normalize_image(image)
            data[2] = image
        if (im_type == 'flair' or im_type == 'fla') and modalities[3]:
            image = normalize_image(image)
            data[3] = image
        if im_type == 'tumor' or im_type == 'seg':
            labels = preprocess_labels(image)

    # remove index where modality is not used
    data = [item for item in data if item is not None]
    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    # random flip around sagittal view
    if not is_test:
        flip = np.random.random()
        if flip < 0.5:
            data = data[:, ::-1, :, :]
            labels = labels[:, ::-1, :]

    return data, labels


def load_data_rembrandt(patient_path, is_test, modalities):
    data = [None] * 4
    patient_path = patient_path.decode('utf-8')

    for im_name in os.listdir(patient_path):
        im_path = os.path.join(patient_path, im_name)
        # for Rembrandt
        im_type = im_name.split('.')[0]
        image = im_path_to_arr(im_path)
        if im_type == 't1' and modalities[0]:
            image = normalize_image(image)
            data[0] = image
        if (im_type == 't1c' or im_type == 't1ce') and modalities[1]:
            image = normalize_image(image)
            data[1] = image
        if im_type == 't2' and modalities[2]:
            image = normalize_image(image)
            data[2] = image
        if (im_type == 'flair' or im_type == 'fla') and modalities[3]:
            image = normalize_image(image)
            data[3] = image
        if im_type == 'tumor' or im_type == 'seg':
            labels = preprocess_labels(image)

    # remove index where modality is not used
    data = [item for item in data if item is not None]
    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    # random flip around sagittal view
    if not is_test:
        flip = np.random.random()
        if flip < 0.5:
            data = data[:, ::-1, :, :]
            labels = labels[:, ::-1, :]

    return data, labels


def train_data_iter(all_patients, patch_size, batch_size, nb_batches, ratio, modalities):
    batch_count = 0
    half_patch = patch_size // 2
    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []
    path_batch = []

    # shuffle order of patients to make sure to mix HGG with LGG
    np.random.shuffle(all_patients)

    for patient_path in all_patients:

        data, labels = load_data_brats(patient_path, False, modalities)

        trimmed_data = data[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch, :]
        trimmed_labels = labels[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch]

        bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
        enhanced = np.where((trimmed_labels == 3) & (trimmed_data[..., 0] != 0))
        necrotic = np.where((trimmed_labels == 1) & (trimmed_data[..., 0] != 0))
        edema = np.where((trimmed_labels == 2) & (trimmed_data[..., 0] != 0))
        fg = np.where((trimmed_labels > 0) & (trimmed_data[..., 0] != 0))

        num_bg = len(bg[0])
        num_enhanced = len(enhanced[0])
        num_necrotic = len(necrotic[0])
        num_edema = len(edema[0])
        num_fg = len(fg[0])

        # oversampling
        assert(len(ratio) == 4), 'you should provide 4 values of ratio for the 4 parts of the tumor'
        assert(np.sum(ratio) == 1), 'the sum of the ratios should be 1'
        ratio_non_tumorous = ratio[0]
        ratio_enhanced = ratio[1]
        ratio_necrotic = ratio[2]

        for _ in range(nb_batches):
            for _ in range(batch_size):
                epsilon = np.random.rand()
                if epsilon <= ratio_non_tumorous:
                    idx = np.random.randint(num_bg)
                    i = bg[0][idx]
                    j = bg[1][idx]
                    k = bg[2][idx]
                elif epsilon <= ratio_enhanced:
                    if num_enhanced == 0:
                        idx = np.random.randint(num_fg)
                        i = fg[0][idx]
                        j = fg[1][idx]
                        k = fg[2][idx]
                    else:
                        idx = np.random.randint(num_enhanced)
                        i = enhanced[0][idx]
                        j = enhanced[1][idx]
                        k = enhanced[2][idx]
                elif epsilon <= ratio_necrotic:
                    if num_necrotic == 0:
                        idx = np.random.randint(num_fg)
                        i = fg[0][idx]
                        j = fg[1][idx]
                        k = fg[2][idx]
                    else:
                        idx = np.random.randint(num_necrotic)
                        i = necrotic[0][idx]
                        j = necrotic[1][idx]
                        k = necrotic[2][idx]
                else:
                    if num_edema == 0:
                        idx = np.random.randint(num_fg)
                        i = fg[0][idx]
                        j = fg[1][idx]
                        k = fg[2][idx]
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


def test_data_iter(all_patients, patch_size, center_size, batch_size, modalities):
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

        data, labels = load_data_brats(patient_path, True, modalities)
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


def get_dataset(directory, is_test, config):
    patients = os.listdir(directory)
    patients = [os.path.join(directory, pat) for pat in patients]
    # need to encode in bytes to pass it to tf.py_func
    # TODO: check if still useful now that we use dataset.from_generator
    patients = [pat.encode('utf-8') for pat in patients]

    patch_size = config.patch_size
    batch_size = config.batch_size
    ratio = config.ratio
    modalities = (config.use_t1pre, config.use_t1post, config.use_t2, config.use_flair)

    if not is_test:
        nb_batches = config.num_train_batches
        def gen():
            return train_data_iter(patients, patch_size, batch_size, nb_batches, ratio, modalities)
        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.string, tf.int32, tf.int32,\
                                                               tf.int32, tf.float32, tf.int32))
        dataset = dataset.apply(tf.contrib.data.unbatch())
        dataset = dataset.shuffle(buffer_size=2000)
    else:
        center_size = config.center_patch
        def gen():
            return test_data_iter(patients, patch_size, center_size, batch_size, modalities)

        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.string, tf.int32, tf.int32,\
                                                               tf.int32, tf.float32, tf.int32))
        dataset = dataset.apply(tf.contrib.data.unbatch())
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset


def data_iter_single_example(patient_path, patch_size, center_size, batch_size):
    data = [None] * 4
    patient_path = patient_path.decode('utf-8')

    for im_name in os.listdir(patient_path):
        im_path = os.path.join(patient_path, im_name)
        # for Brats2017
        im_type = im_name.split('.')[0].split('_')[-1]
        if im_type == 't1':
            image = im_path_to_arr(im_path)
            data[0] = normalize_image(image)
        if im_type == 't1c' or im_type == 't1ce':
            image = im_path_to_arr(im_path)
            data[1] = normalize_image(image)
        if im_type == 't2':
            image = im_path_to_arr(im_path)
            data[2] = normalize_image(image)
        if im_type == 'flair' or im_type == 'fla':
            image = im_path_to_arr(im_path)
            data[3] = normalize_image(image)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    patient_path = patient_path.encode('utf-8')

    batch_count = 0
    half_patch = patch_size // 2
    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []
    path_batch = []

    i_len, j_len, k_len = (155, 240, 240)

    for i in get_patch_centers_fcn(i_len, patch_size, center_size):
        for j in get_patch_centers_fcn(j_len, patch_size, center_size):
            for k in get_patch_centers_fcn(k_len, patch_size, center_size):

                x = data[i - half_patch:i + half_patch,\
                         j - half_patch:j + half_patch,\
                         k - half_patch:k + half_patch, :]
                y = np.zeros((patch_size, patch_size, patch_size))

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


def get_dataset_single_patient(patient, batch_size, patch_size, center_size):
    def gen():
        return data_iter_single_example(patient, patch_size, center_size, batch_size)
    dataset = tf.data.Dataset.from_generator(generator=gen,
                                             output_types=(tf.string, tf.int32, tf.int32,\
                                                           tf.int32, tf.float32, tf.int32))
    dataset = dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.batch(batch_size)
    return dataset
