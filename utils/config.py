class Config:

    def __init__(self, path):
        param_dict = {}
        with open(path, 'r') as f:
            for line in f:
                param_dict[line.split()[0]] = line.split()[1]
        
        self.res_path = param_dict.get('res_path')
        self.ckpt_path = param_dict.get('ckpt_path')
        
        self.train_path = param_dict.get('train_path')
        self.val_path = param_dict.get('val_path')

        # learning rate schedule parameters
        # self.lr = float(param_dict.get('lr'))
        self.lr_init = float(param_dict.get('lr_init'))
        self.lr_min = float(param_dict.get('lr_min'))
        self.start_decay = float(param_dict.get('start_decay'))  # id of epoch to start decay
        self.decay_rate = float(param_dict.get('decay_rate'))  # id of epoch to end decay
        self.end_decay = float(param_dict.get('end_decay'))
        self.lr_warm = float(param_dict.get('lr_warm'))
        self.lr_warm = float(param_dict.get('lr_warm'))
        self.end_warm = float(param_dict.get('end_warm'))

        self.l2 = float(param_dict.get('l2'))

        self.batch_size = int(param_dict.get('batch_size'))
        self.num_epochs = int(param_dict.get('num_epochs'))
        self.num_train_batches = int(param_dict.get('num_train_batches'))
        self.num_val_batches = int(param_dict.get('num_val_batches'))

        self.fine_tune_ckpt_path = param_dict.get('fine_tune_ckpt_path')
