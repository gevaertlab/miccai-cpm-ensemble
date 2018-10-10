import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit  
from PIL import Image



class Dataset:
    
    def __init__(self, config):
            self.config= config
            self._train_val_dir = "/labs/gevaertlab/data/MICCAI/patches_448"
            self._test_dir = "/labs/gevaertlab/data/MICCAI/patches_448_test"
            self.le = LabelEncoder()
            self._partition = self.get_partition()


    def get_binarized_data(self):

        df = pd.read_table('MICCAI_labels.txt', index_col = 0, delim_whitespace = True, header = 0)
        df = df.apply(self.le.fit_transform)
        return df

    def get_partition(self):
        
        df = self.get_binarized_data()
        ids = df.index 
        labels = df.values.flatten()
      
        sss = StratifiedShuffleSplit(n_splits=1, test_size= self.config.val_size)
        sss.get_n_splits(ids, labels)
        for train_index, test_index in sss.split(ids, labels):
            ids_train, ids_val = ids[train_index], ids[test_index]
            y_train, y_val = labels[train_index], labels[test_index]    
        
        test_data = pd.read_table('MICCAI_Test.txt', index_col = 0, delim_whitespace = True, header = 0)

        ids_test = test_data.index
        y_test = test_data.apply(self.le.fit_transform).values.flatten()

        partition_ids = {'train': list(ids_train), 'val': list(ids_val), 'test': list(ids_test)}        
        partition_labels = {'train': list(y_train), 'val': list(y_val), 'test': list(y_test)}
    
        return partition_ids, partition_labels  

    def convert_to_arrays(self, samples, labels,  phase = ['train','val','test'], size = 1):
        
        if phase == 'test':
            directory = self._test_dir

        else: 
             directory = self._train_val_dir
                 
        X, ids = [], []
        for sample in samples:
            patches = os.listdir(directory + "/%s" %sample)
            patches = np.random.choice(patches, size= size, replace=True)
            for patch in patches:
                ID = directory + "/%s/%s"% (sample, patch)
                ids.append(ID)
                img = Image.open(ID)
                img = img.resize((224, 224))
                image = np.array(img)[:,:,:3]
                X.append(image)  
        X = np.asarray(X)
        y = np.repeat(labels, size)
        return X, y