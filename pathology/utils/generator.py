import numpy as np
from PIL import Image


class Generator(object):
    
    def __init__(self, config, dataset):
        
        self.config = config
        self.dataset = dataset
        self.list_IDs = self.dataset._partition[0]['train']
        self.list_labels = self.dataset._partition[1]['train']
    
    def generate(self):
        'Generates batches of samples'
        
        while 1:
            indexes = self.__get_exploration_order()
            imax = int(len(indexes)/self.config.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.list_IDs[k] for k in indexes[i*self.config.batch_size:(i+1)*self.config.batch_size]]
                list_labels_temp = [self.list_labels[k] for k in indexes[i*self.config.batch_size:(i+1)*self.config.batch_size]]

                X, y = self.__data_generation(list_IDs_temp,list_labels_temp)

                yield X, y

    def __get_exploration_order(self):
        'Generates order of exploration'
        indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        
        X, y = self.dataset.convert_to_arrays(list_IDs_temp, list_labels_temp, size = self.config.sampling_size_train)
        
        return X, y
    
    
  
    
    
    