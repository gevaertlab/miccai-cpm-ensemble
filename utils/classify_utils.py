import os
import re

import dicom


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
def find_modality(description):
    description = description.lower()
    modalities = [x for x in MODALITIES if x in description]
    if len(modalities) == 0:
        modality = 'N/A'
    elif len(modalities) == 1:
        if 't1' in modalities:
            if 'pre' in description:
                modality = 't1pre'
                
            elif any(re.search('(?<![a-zA-Z])' + x + '(?![a-zA-Z])', description)\
                     for x in ['post', 't1c', '\+c', '\+ c', 'c\+', 'con', 'gd', 'gad']):
                modality = 't1post'
            else:
                modality = 't1 pre or post?'
        else:
            modality = modalities[0]
    else:
        if set(modalities) == set(['flair', 't2']):
            modality = 'flair'
        else:
            modality = 'multiple modalities detected'

    return modality


def collect_info_patient_folder(patient_folder):  
    dicoms = os.listdir(patient_folder)
    dicoms = [dic for dic in dicoms if dic[-4:] == '.dcm']
    sample = dicoms[0]
    sample = os.path.join(patient_folder, sample)
    sample = dicom.read_file(sample)

    # collect information
    patient_id = str(sample.PatientName)
    date = str(sample.AcquisitionDate)
    thickness = str(sample.SliceThickness)
    rows = int(sample.Rows)
    columns = int(sample.Columns)
    nb_dicoms = len(dicoms)
    description = str(sample.SeriesDescription)
    view = find_view(description)
    modality = find_modality(description)

    return patient_id, view, modality, date, thickness, rows, columns, nb_dicoms, description