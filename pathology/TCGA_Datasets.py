import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit  
from PIL import Image


class TCGA_Dataset:
    
    def __init__(self, config):

        self.config = config
        self._train_val_dir = '/labs/gevaertlab/data/cedoz/patches_448'
        self._test_dir = '/labs/gevaertlab/data/MICCAI/patches_448_test'
       # self._test_dir = '/labs/gevaertlab/data/MICCAI/patches_448'
        self.le = LabelEncoder()
        self._samples = self.get_samples()
        self._labels  = self.get_labels()
        self._partition = self.get_partition()

     
    def get_samples(self):
        
        samples_0 = os.listdir(self._train_val_dir)
        samples_1= list(pd.read_excel('TCGA-MICCAI-Patients.xlsx', index_col = 'Patient').index)
        samples = np.intersect1d(samples_0, samples_1)
        return samples

    
    def get_labels(self):
        df = pd.read_excel('TCGA-MICCAI-Patients.xlsx',index_col = 'Patient')
        df = df[df.index.isin(self._samples)]
        labels = df.apply(self.le.fit_transform).values.flatten()
        le_name_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        print(le_name_mapping)
        return labels
  
    def get_partition(self):
        
        df = pd.read_excel('TCGA-MICCAI-Patients.xlsx',index_col = 'Patient')
        df = df[df.index.isin(self._samples)]
        ids = df.index
        labels = self._labels
 
        sss = StratifiedShuffleSplit(n_splits=1, test_size= self.config.val_size)
        sss.get_n_splits(ids, labels)
        for train_index, test_index in sss.split(ids, labels):
            ids_train, ids_val = ids[train_index], ids[test_index]
            y_train, y_val = labels[train_index], labels[test_index]    
        
        test_data = pd.read_table('MICCAI_Test.txt', index_col = 0, delim_whitespace = True, header = 0)
        #test_data = pd.read_table('MICCAI_Test.txt', index_col = 0, delim_whitespace = True, header = 0)        

        ids_test = test_data.index
        y_test = test_data.apply(self.le.fit_transform).values.flatten()
        partition_ids = {'train': list(ids_train), 'val': list(ids_val), 'test': list(ids_test)}
      #  partition_labels = {'train': list(y_train), 'val': list(y_val)}
        
        partition_labels = {'train': list(y_train), 'val': list(y_val), 'test': list(y_test)}
    
        return partition_ids, partition_labels 
    

    def convert_to_arrays(self, samples, labels, phase = ['train','val','test'], size = 1):
        
        if phase == 'test':
            directory = self._test_dir
        else: 
            directory = self._train_val_dir
                
        X, ids = [], []
        for sample in samples:
            patches = os.listdir(directory + '/%s' % sample)
            patches = np.random.choice(patches, size= size, replace=True)
            for patch in patches:
                ID = directory + "/%s/%s"% (sample, patch)
                ids.append(ID)               
                img = Image.open(ID)
                img = img.resize((224,224))
                image = np.array(img)[:,:,:3]
                X.append(image)  
        X = np.asarray(X)        
        y = np.repeat(labels, size)
        
        return X, y