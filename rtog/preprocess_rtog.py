import re
import csv


raw_files_path = 'rtog_files.txt'

types_of_scan = ['MR', 'CT']
AXIAL_VIEWS = ['ax', 'axial']
SAGITAL_VIEWS = ['sag', 'sagital']
CORONAL_VIEWS = ['cor', 'coronal']
MODALITIES = ['t1', 't2', 'flair', 'dwi', 'gre', 'blade', 'propeller', 'lava', 'fame', 'mprage']
# mprage instead of rage, otherwise match on 'average'


def find_view(raw_file_name):
    if any(x in raw_file_name for x in AXIAL_VIEWS):
        view = 'axial'
    elif any(x in raw_file_name for x in SAGITAL_VIEWS):
        view = 'sagital'
    elif any(x in raw_file_name for x in CORONAL_VIEWS):
        view = 'coronal'
    else:
        view = 'N/A'
    return view


def find_modality(raw_file_name):
    modalities = [x for x in MODALITIES if x in raw_file_name]
    if len(modalities) == 0:
        modality = 'N/A'
    elif len(modalities) == 1:
        if 't1' in modalities:
            if 'pre' in raw_file_name:
                modality = 't1pre'
            elif any(x in raw_file_name for x in ['post', 't1c']):
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


def classify_patient_file(raw_file_name):
    patient_id = 'N/A'
    view = 'N/A'
    modality = 'N/A'

    if 'MR' in raw_file_name:
        tokens = raw_file_name.split('MR')
    elif 'CT' in raw_file_name:
        tokens = raw_file_name.split('CT')
    else:
        return patient_id, view, modality

    if re.match(r'(\d+)(\^)(\d+)(\^)(\d+)(\_)(\d+)', tokens[0]):
        patient_id = tokens[0]
        truncate_pos = len(patient_id) + len('MR')
        truncated_path = raw_file_name[truncate_pos:]
        truncated_path = truncated_path.lower()
        view = find_view(truncated_path)
        modality = find_modality(truncated_path)
    return patient_id, view, modality


with open(raw_files_path, 'r') as fin:
    all_files = fin.readlines()
    all_files = [x.strip() for x in all_files]

with open('rtog_patients.csv', 'w') as csvfile:
    fieldnames = ['study', 'raw_name', 'curated_id', 'curated_view', 'curated_modality']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for file in all_files:
        study = file.strip().split('/')[0]
        raw_file = file.strip().split('/')[1]
        curated_id, curated_view, curated_modality = classify_patient_file(raw_file)
        writer.writerow({'study': study, 'raw_name': raw_file, 'curated_id': curated_id,
                         'curated_view': curated_view, 'curated_modality': curated_modality})


"""
Problems:
- several cases present 2 modalities:  '79^154^825_79_PR_2009-08-12_102200_MRI.BRAIN.W..+.W.O.CONTRAST_AXIAL.T2.FLAIR.CLEAR_n1__00000' ---> FLAIR
- sometimes, no modality:  '29^9521^825_29_MR_2009-08-02_231136_HEAD..MRI^ROUTINE.HEAD_AX.GRE.HEMO_n24__00000'
- sometimes, no view:  '3^131^825_3_MR_2009-08-24_105043_BRAIN^MS_t1.mp.RAGE.POST.CONT.1.0mm_n160__00000'  ---> seems to be axial
- sometimes, no indication at all:  '56^2424^825_56_MR_2009-08-16_083617_MRI.BRAIN.W.WO_ep2d.diff.3scan.trace_n78__00000'
- AX + DWI ? :  '173^1324^825_173_MR_2009-11-17_215757_MRI.BRAIN.WO&W.CONT.70553_AX.DWI-not.prop_n64__00000' ---> separate
- Axial + GRE? :  '183^7631^825_183_MR_2009-11-27_154540_MRI.BRAIN.W.WO.CONTRAST_.AXIAL.GRE.GE_n24__00000' ---> T2 if no other calls 
- Several T2 modalities for same patient:
    - 90^9521^825_90_MR_2009-11-17_163549_HEAD..MRI^ROUTINE.HEAD_AX.T2.BLADE_n22__00000 ---> T2
    - 90^9521^825_90_MR_2009-11-17_163549_HEAD..MRI^ROUTINE.HEAD_AX.IR.T2.BLADE_n22__00000
"""
