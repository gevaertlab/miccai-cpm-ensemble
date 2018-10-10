import os
import re

import pydicom


types_of_scan = ['MR', 'CT', 'PR']
AXIAL_VIEWS = ['axl', 'axial', 'ax']
SAGITAL_VIEWS = ['sag', 'sagital']
CORONAL_VIEWS = ['cor', 'coronal']
MODALITIES = ['t1', 't2', 'flair', 'dwi', 'gre', 'blade', 'propeller', 'lava', 'fame', 'mprage']
# mprage instead of rage, otherwise match on 'average'


def find_view(description):
    description = description.lower()
    if any(re.search('(?<![a-zA-Z])' + x + '(?![a-zA-Z])', description) for x in AXIAL_VIEWS):
        view = 'axial'
    elif any(re.search('(?<![a-zA-Z])' + x + '(?![a-zA-Z])', description) for x in SAGITAL_VIEWS):
        view = 'sagital'
    elif any(re.search('(?<![a-zA-Z])' + x + '(?![a-zA-Z])', description) for x in CORONAL_VIEWS):
        view = 'coronal'
    else:
        view = 'N/A'
    return view

# TODO: look at Echo Time and Inversion Time fields in the metadata to classify between T1 and T2
# Candidate T1posts:
# PG?
# SE?
# PosDisp: [11] t1_mpr_axial  2mmGAD?

# BLADE?
# MPR?
# MTC?
# MPRAGE?
# Propeller?
# Subtelty between T1Flair and T2Flair?
def find_modality(description):
    description = description.lower()
    recognized_modalities = [x for x in MODALITIES if x in description]
    if len(recognized_modalities) == 0:
        modality = 'N/A'

    elif len(recognized_modalities) == 1:
        if 't1' in recognized_modalities:
            if 'pre' in description:
                modality = 't1pre'

            elif any(re.search('(?<![a-zA-Z])' + x + '(?![a-zA-Z])', description) \
                     for x in ['post', 't1c', 'con', 'postc',
                               'gd', 'gad', 'gado', 'magnevist', 'gadavist',
                               'mtc', 'mpr']):
                modality = 't1post'
            elif any(re.search(x + '(?![a-zA-Z])', description) \
                     for x in ['\+c', '\+ c']):
                modality = 't1post'
            elif any(re.search('(?<![a-zA-Z])' + x, description) \
                     for x in ['c\+', 'c \+']):
                modality = 't1post'
            else:
                modality = 't1 pre or post?'
        else:
            modality = recognized_modalities[0]

    else:
        if set(recognized_modalities) == set(['flair', 't2']):
            modality = 'flair'
        else:
            modality = 'multiple modalities detected: ' + "/".join(recognized_modalities)

    return modality


def collect_info_patient_folder(patient_folder):
    dicoms = os.listdir(patient_folder)
    dicoms = [dic for dic in dicoms if dic[-4:] == '.dcm']
    sample = dicoms[0]
    sample = os.path.join(patient_folder, sample)
    sample = pydicom.read_file(sample)

    # collect information
    patient_id = str(sample.PatientName)

    try:
        date = str(sample.AcquisitionDate)
    except AttributeError:
        date = 'N/A'

    try:
        thickness = str(sample.SliceThickness)
    except AttributeError:
        thickness = 'N/A'

    try:
        rows = int(sample.Rows)
    except:
        rows = -1

    try:
        columns = int(sample.Columns)
    except:
        columns = -1

    nb_dicoms = len(dicoms)

    try:
        description = str(sample.SeriesDescription)
    except:
        description = 'N/A'

    view = find_view(description)
    modality = find_modality(description)

    return patient_id, view, modality, date, thickness, rows, columns, nb_dicoms, description