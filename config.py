class Config(object):

    def __init__(self, data_path="/labs/gevaertlab/data/MICCAI/pathology", patch_size=448, threshold=0.4,
                 selected_features=['out'], input_shape = 224, val_size = 0.30, test_size = 0.00, epochs = 30, gpu = "0", sampling_size_train  = 10, sampling_size_val = 10, sampling_size_test = 500, batch_size = 16, lr = 5e-6, lr_decay=1e-6, from_idx=0):
        
        self.data_path = data_path
        self.patch_size = patch_size
        self.threshold = threshold
        self.selected_features = selected_features
        self.input_shape = input_shape
        self.test_size = test_size
        self.val_size = val_size
        self.epochs = epochs
        self.gpu = gpu
        
        self.sampling_size_train = sampling_size_train
        self.sampling_size_val = sampling_size_val 
        self.sampling_size_test = sampling_size_test

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.from_idx = from_idx
        
