import os
from tqdm import tqdm
data_dir = '/labs/gevaertlab/users/hackhack/RTOG/scratch'

with open('rtog/outputs/rtog_files.txt', 'w') as f:
    for study in tqdm(os.listdir(data_dir)):  
        if 'Studies' in study:
            study_path = os.path.join(data_dir, study)
            for filename in os.listdir(study_path):
                filename_path = os.path.join(study_path, filename)
                if len(os.listdir(filename_path)) > 0:
                    full_name = study + '/' + filename
                    f.write(full_name)
                    f.write('\n') 
