import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
import os
import numpy as np
from config import Config
from PIL import Image


class Dataset:
    
    def __init__(self, config):
            self.config= config

    def get_binarized_data(self):

      #  output_data = pd.read_table('MICCAI_labels.txt', index_col = 0, delim_whitespace = True, header = 0)
      #  binarized_data = output_data.apply(lambda x: LabelBinarizer().fit_transform(x)[:, 0], axis=0)
        return binarized_data

    def get_labels(self):
        samples = os.listdir("/labs/gevaertlab/data/MICCAI/patches_448")
        labels = {}
        data = self.get_binarized_data()
        for i in data.columns:
            labels[i] = {}
            for s in samples:
                labels[i][s] = data.loc[s,i]
        return labels

    def get_ids(self, samples):

        ids = []
        for sample in samples:
            patches = os.listdir("/labs/gevaertlab/data/MICCAI/patches_448/%s" % sample)
            patches = np.random.choice(patches, size= self.config.sampling_size_train, replace=True)
            for patch in patches:
                ID = "/labs/gevaertlab/data/MICCAI/patches_448/%s/%s"%(sample, patch)
                ids.append(ID)
        return ids


    def get_partition():

        samples = os.listdir("/labs/gevaertlab/data/MICCAI/patches_448")
        np.random.shuffle(samples)
        idx_val = int((1- self.config.val_size)*len(samples))
        idx_test = int((1 - self.config.test_size) * len(samples))
        train_samples, val_samples, test_samples = np.split(samples, [idx_val, idx_test])
        train_samples, val_samples, test_samples = list(train_samples), list(val_samples), list(test_samples)
        train_ids = get_ids(train_samples)
        partition = {'train': train_samples, 'val': val_samples, 'test': test_samples}
        return partition

    def convert_to_arrays(samples, labels):

            X, ids = [], []
            for sample in samples:
                patches = os.listdir("/labs/gevaertlab/data/MICCAI/patches_448/%s"%sample)
                patches = np.random.choice(patches, size= 80, replace=True)
                for patch in patches:
                    ID = "/labs/gevaertlab/data/MICCAI/patches_448/%s/%s"% (sample, patch)
                    ids.append(ID)
                    img = Image.open(ID)
                    img = img.resize((224, 224))
                    image = np.array(img)[:,:,:3]
                    X.append(image)  
            X = np.asarray(X)

            for label in labels.keys():
                y_label = []
                for ID in ids:
                    sample = ID.split('/')[-2]
                    y_label.append(labels[label][sample])
                y = np.asarray(y_label)

            return X, y