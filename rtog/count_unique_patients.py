import os
from tqdm import tqdm 

data_dir = '/local-scratch/marcthib_scratch/rtog_raw'

count_scan_folders = 0
count_nonempty_scan_folders = 0
unique_patients = set()
unique_patients_study = set()

for study in tqdm(os.listdir(data_dir)):  
    if 'Studies' in study:
        study_path = os.path.join(data_dir, study)
        for filename in os.listdir(study_path):
            count_scan_folders += 1
            filename_path = os.path.join(study, filename)
              
            if 'MR' in filename:
                tokens = filename.split('MR')
            elif 'CT' in filename:
                tokens = filename.split('CT')
            else:
                tokens = ['N/A']
            patient_id = tokens[0]
            unique_patients.add(patient_id)
            
            if 'MR' in filename_path:
                tokens = filename_path.split('MR')
            elif 'CT' in filename:
                tokens = filename_path.split('CT')
            else:
                tokens = ['N/A']
            patient_study_id = tokens[0]
            unique_patients_study.add(patient_study_id)
            

print("Found {} scan folders".format(count_scan_folders))
print("Out of which  {} are non-empty.".format(count_nonempty_scan_folders))

print("Found {} unique patients".format(len(unique_patients)))
print("Found {} unique patients + study".format(len(unique_patients_study)))

