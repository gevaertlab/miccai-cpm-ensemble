import os

import numpy as np
import tensorflow as tf

from radiology.utils.data_utils import im_path_to_arr, resize_raw_to_base
from radiology.utils.data_utils import normalize_image

KEPT_PATIENTS = set(["cbtc_train_" + str(i) for i in range(33) if i not in [1, 8, 9, 18, 26]])
KEPT_PATIENTS_TEST = set(["cbtc_test_" + str(i) for i in range(33) if i not in [31]])


def load_data_miccai(patient_path, is_test, modalities):
    data = [None] * len(modalities)
    patient_path = patient_path.decode('utf-8')

    im_type_to_path = {}
    for im_name in os.listdir(patient_path):
        if "proc" not in im_name:
            continue
        im_path = os.path.join(patient_path, im_name)
        im_type = im_name.split('_')[0].lower()
        im_type_to_path[im_type] = im_path

    for im_type in im_type_to_path:
        image = im_path_to_arr(im_type_to_path[im_type])
        image = resize_raw_to_base(image)
        if im_type == 't1c' and modalities[0]:
            image = normalize_image(image)
            data[0] = image
        elif im_type == 'flair' and modalities[1]:
            image = normalize_image(image)
            data[1] = image

    # remove index where modality is not used
    data = [item for item in data if item is not None]
    # data = [resize_data_to_brats_size(item) for item in data]
    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    # random flip around sagittal axis
    if not is_test:
        flip = np.random.random()
        if flip < 0.5:
            data = data[:, :, ::-1, :]

    return data


def gen_tcga_miccai(directory, is_test, config):
    modalities = (config.use_t1post, config.use_flair, config.use_segmentation)

    patients = os.listdir(directory)
    patients = list(set(patients).intersection(KEPT_PATIENTS.union(KEPT_PATIENTS_TEST)))
    patients = [os.path.join(directory, pat) for pat in patients]
    patients = [pat.encode('utf-8') for pat in patients]  # need to encode in bytes to pass it to tf.py_func

    patients.sort()
    if not is_test:
        np.random.seed(0)
        np.random.shuffle(patients)

    patients_stage_path = os.path.join(directory,
                                       "../datasets_None_4b87ae5a-4ca7-4b95-99f5-09ce31da60e0_README_all_training.txt")
    with open(patients_stage_path, "r") as f:
        lines = f.readlines()
    labels = {int(q.split(" ")[0].split("_")[-1]): {'A': 1, 'O': 0}[q.strip("\n")[-1]] for q in lines[10:]}

    for patient in patients:
        image = load_data_miccai(patient, is_test, modalities)
        patient_id = int(patient.decode("utf-8").split("/")[-1].split("_")[-1])
        if patient_id in labels:
            stage = labels[patient_id]
        else:
            stage = -1
        print(patient_id)
        yield image, stage, patient_id


def get_dataset_batched(directory, is_test, config):
    def gen():
        return gen_tcga_miccai(directory, is_test, config)

    dataset = tf.data.Dataset.from_generator(generator=gen,
                                             output_types=(tf.float32, tf.float32, tf.int32))
    batch_size = config.batch_size
    batched_dataset = dataset.batch(batch_size)
    batched_dataset = batched_dataset.prefetch(1)

    return batched_dataset
