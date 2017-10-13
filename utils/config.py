class Config:

    def __init__(self, path):
        param_dict = {}
        with open(path, 'rb') as f:
            for line in f:
                param_dict[line.split()[0]] = line.split()[1]
        
        self.res_path = param_dict['res_path']
        self.ckpt_path = param_dict['ckpt_path']
        
        self.train_path = param_dict['train_path']
        self.val_path = param_dict['val_path']

        self.lr = float(param_dict['lr'])
        self.l2 = float(param_dict['l2'])

        self.batch_size = int(param_dict['batch_size'])
        self.num_epochs = int(param_dict['num_epochs'])
        self.num_train_batches = int(param_dict['num_train_batches'])
        self.num_val_batches = int(param_dict['num_val_batches'])

        self.fine_tune_ckpt_path = param_dict['fine_tune_ckpt_path']
