import numpy as np
import tensorflow as tf

from models.model import Model

from utils.data_utils import get_ex_paths, get_shape_v2
from utils.dice_score import dice_score


class FCN_Model(Model):

    def __init__(self, config):
        self.config = config
        self.patch = config.patch_size
        self.nb_classes = config.nb_classes
        self.nb_modalities = config.use_t1pre + config.use_t1post + config.use_t2 + config.use_flair

        self.load_data()
        self.add_dataset()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def load_data(self):
        self.train_ex_paths = get_ex_paths(self.config.train_path)
        self.val_ex_paths = get_ex_paths(self.config.val_path)

    def add_dataset(self):
        raise NotImplementedError

    def add_placeholders(self):
        self.image_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, self.patch, self.patch, self.patch, 4])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                shape=[None, self.patch, self.patch, self.patch])
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[])
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])
        self.is_training = tf.placeholder(tf.bool, shape = [])

        # for tensorboard
        tf.summary.scalar("lr", self.lr_placeholder)

    def add_model(self):
        raise NotImplementedError

    def add_pred_op(self):
        probs = tf.nn.softmax(tf.reshape(self.score, [-1, self.nb_classes]))
        reshape_probs = tf.reshape(probs, tf.shape(self.score))

        self.pred = tf.argmax(reshape_probs, 4)
        self.prob = reshape_probs

    def add_loss_op(self):
        raise NotImplementedError

    def get_variables_to_restore(self, level=3):
        raise NotImplementedError

    def add_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                 .minimize(self.loss, global_step=self.global_step)

            if self.config.finetuning_method == 'last_layers':
                var_to_train, _ = self.get_variables_to_restore(self.config.finetuning_level)
                self.train_last_layers = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                                 .minimize(self.loss, var_list=var_to_train,\
                                                           global_step=self.global_step)
