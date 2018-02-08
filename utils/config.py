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

        # model architecture
        self.kernel_size = int(param_dict.get('kernel_size', 5))
        self.nb_classes = int(param_dict.get('nb_classes', 2))
        self.nb_filters = int(param_dict.get('nb_filters', 10))
        self.use_t1pre = param_dict.get('use_t1pre', 'True') == 'True'
        self.use_t1post = param_dict.get('use_t1post', 'True') == 'True'
        self.use_t2 = param_dict.get('use_t2', 'True') == 'True'
        self.use_flair = param_dict.get('use_flair', 'True') == 'True'

        # learning rate schedule
        self.lr_init = float(param_dict.get('lr_init', 1e-4))
        self.lr_min = float(param_dict.get('lr_min', 1e-6))
        self.start_decay = float(param_dict.get('start_decay', 10))  # id of epoch to start decay
        self.decay_rate = float(param_dict.get('decay_rate', 0.8))
        self.end_decay = float(param_dict.get('end_decay', 30))  # id of epoch to end decay
        self.lr_warm = float(param_dict.get('lr_warm', 5e-5))
        self.end_warm = float(param_dict.get('end_warm', 3))

        # regularization
        self.l2 = float(param_dict.get('l2', 1e-4))
        self.dropout = float(param_dict.get('dropout', 0.5))
        self.use_batch_norm = param_dict.get('use_batch_norm', 'False') == 'True'

        # data sampling
        self.patch_size = int(param_dict.get('patch_size', 32))
        self.center_patch = int(param_dict.get('center_patch', 10))
        self.ratio = [float(param_dict.get(x, '0.25')) for x in ['non_tumorous', 'enhanced', 'necrotic', 'edema']]
        self.batch_size = int(param_dict.get('batch_size', 50))
        self.num_train_batches = int(param_dict.get('num_train_batches', 20))
        self.num_val_batches = int(param_dict.get('num_val_batches', 20))
        self.num_epochs = int(param_dict.get('num_epochs', 50))

        # loss
        self.use_mask = param_dict.get('use_mask', 'False') == 'True'
        self.use_dice_whole_loss = param_dict.get('use_dice_whole_loss', 'False') == 'True'
        self.use_dice_core_loss = param_dict.get('use_dice_core_loss', 'False') == 'True'
        self.use_dice_enhancing_loss = param_dict.get('use_dice_enhancing_loss', 'False') == 'True'
        self.ds_loss_beta = float(param_dict.get('ds_loss_beta', 0.5))

        # finetuning
        self.ckpt_path_to_finetune = param_dict.get('ckpt_path_to_finetune')
        self.finetuning_method = param_dict.get('finetuning_method', 'all_layers')