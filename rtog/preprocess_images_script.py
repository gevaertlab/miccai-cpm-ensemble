#! /usr/bin/env python

EXCLUDE_LIST = ['121', '175', '466', '488', '526', '589', '809', '853', '919']

import getpass
import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import im_path_to_arr
from utils.process_rtog_nii import process_rtog_nii

def run_preprocessing(scratch_folder, patient_number):
    process_rtog_nii(scratch_folder,
                     input_t1c_filename=patient_number + 't1c.nii',
                     output_t1c_filename=patient_number + 't1c_proc.nii',
                     input_flair_filename=patient_number + 'flair.nii',
                     output_flair_filename=patient_number + 'flair_proc.nii')

def viz_preprocessing_results(scratch_folder, patient_number):
    fig, axes = plt.subplots(4, 4)
    for i, imtype in enumerate(['t1c', 't1c_proc', 'flair', 'flair_proc']):
        impath = os.path.join(scratch_folder, str(patient_number) + imtype + '.nii')
        arr = np.nan_to_num(im_path_to_arr(impath))
        step = int(len(arr) / 4)
        idxs = [int(step / 2 + sl * step) for sl in range(4)]
        for k, idx in enumerate(idxs):
            ax = axes[k, i]
            ax.imshow(arr[idx], cmap='viridis', vmin=0, vmax=arr.max())
            ax.axis('off')
    plt.tight_layout()
    plt.suptitle(str(patient_number))
    plt.show()
    input()
    plt.close()

def main():
    scratch_folder = '/local-scratch/' + getpass.getuser() + '_scratch/rtog'
    # i = 0
    for item in os.listdir(scratch_folder):
        if os.path.isfile(os.path.join(scratch_folder, item)):
            patient_number = re.findall('\d+', item)[0]
            if patient_number in EXCLUDE_LIST:
                continue
            # i += 1
            # if not i % 10:
            #     viz_preprocesing_results(scratch_folder, patient_number)
            if os.path.exists(os.path.join(scratch_folder,
                              patient_number + 't1c_proc.nii')):
                continue
            run_preprocessing(scratch_folder, patient_number)

if __name__ == '__main__':
    main()
