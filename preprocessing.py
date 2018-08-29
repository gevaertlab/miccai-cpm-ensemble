import numpy as np
import os
import shutil
import PIL.ImageOps
from openslide import *
from scipy.ndimage.morphology import binary_dilation
from multiprocessing import Pool
from config import Config
import cv2


def get_patches(patient_id, config):
    if os.path.isdir("/labs/gevaertlab/data/MICCAI/patches_448_test/%s"%patient_id):
        print ("sample already processed")
        return
    if os.path.isdir("/labs/gevaertlab/data/MICCAI/temp/%s"% patient_id):
        shutil.rmtree("/labs/gevaertlab/data/MICCAI/temp/%s"% patient_id)
    os.makedirs("/labs/gevaertlab/data/MICCAI/temp/%s"% patient_id)
    img = OpenSlide("/labs/gevaertlab/data/MICCAI/pathology_test/%s.svs"% patient_id)
    width, height = img.dimensions
    idx = 0
    for i in range(int(height/config.patch_size)):
        print ("iteration %d out of %d"%(i+1,int(height/config.patch_size)))
        for j in range(int(width/config.patch_size)):
            patch = img.read_region(location=(j*config.patch_size,i*config.patch_size), level=0,
                                    size=(config.patch_size,config.patch_size)).convert('RGB')
            array = np.array(patch)[:,:,:3]    
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = binary_dilation(thresh, iterations=15)
            ratio = np.mean(thresh)
            if ret < 200 and ratio > 0.80:
                patch.save("/labs/gevaertlab/data/MICCAI/temp/%s/%s.jpg"% (patient_id, idx))
                idx += 1
    shutil.move("/labs/gevaertlab/data/MICCAI/temp/%s"% patient_id, "/labs/gevaertlab/data/MICCAI/patches_448_test/%s"% patient_id)

def get_all_patches(config, processes=30):
    
    patient_ids = os.listdir("/labs/gevaertlab/data/MICCAI/pathology_test/")
    patient_ids = [patient_id[:-4] for patient_id in patient_ids]    
    p = Pool(processes)
    p.starmap(get_patches, [(patient_id, config) for patient_id in patient_ids])

config = Config()
get_all_patches(config)