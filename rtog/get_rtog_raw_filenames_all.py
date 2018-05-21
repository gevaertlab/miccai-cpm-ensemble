import os
from tqdm import tqdm 

data_dir = '/labs/gevaertlab/users/hackhack/RTOG/scratch'

count_scan_folders = 0
count_nonempty_scan_folders = 0
with open('rtog/outputs/all_rtog_files.txt', 'w') as f:
    for study in tqdm(os.listdir(data_dir)):  
        if 'Studies' in study:
            study_path = os.path.join(data_dir, study)
            for filename in os.listdir(study_path):
                count_scan_folders += 1
                filename_path = os.path.join(study_path, filename)
                if len(os.listdir(filename_path)) > 0:
                    count_nonempty_scan_folders += 1
                full_name = study + '/' + filename
                f.write(full_name)
                f.write('\n')

print("Found {} scan folders".format(count_scan_folders))
print("Out of which  {} are non-empty.".format(count_nonempty_scan_folders))

