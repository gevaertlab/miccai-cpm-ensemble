import numpy as np
import tensorflow as tf

from models.model import Model

from utils.data_iterator import fcn_data_iter_v2
from utils.data_utils import get_ex_paths, get_shape_v2
from utils.dice_score import dice_score


class FCN_Model(Model):

    def __init__(self, config):
        self.config = config
        self.patch = config.patch_size
        self.nb_classes = config.nb_classes

        self.load_data()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def load_data(self):
        self.train_ex_paths = get_ex_paths(self.config.train_path)
        self.val_ex_paths = get_ex_paths(self.config.val_path)

    def add_placeholders(self):
        self.image_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, self.patch, self.patch, self.patch, 4])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                shape=[None, self.patch, self.patch, self.patch])
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[])
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])
        self.is_training = tf.placeholder(tf.bool, shape = [])

    def add_model(self):
        batch_size = tf.shape(self.label_placeholder)[0]

        with tf.variable_scope('conv1'):

            conv1 = tf.layers.conv3d(inputs=self.image_placeholder, filters=10,
                                     kernel_size=[5, 5, 5], padding="SAME",
                                     use_bias=True, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))
            # drop1 = tf.nn.dropout(conv1, self.dropout_placeholder)
            pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            print(conv1.get_shape())
            print(pool1.get_shape())

        with tf.variable_scope('conv2'):

            conv2 = tf.layers.conv3d(inputs=pool1, filters=20,
                                     kernel_size=[5, 5, 5], padding="SAME",
                                     use_bias=True, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))
            # drop2 = tf.nn.dropout(conv2, self.dropout_placeholder)
            pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            print(conv2.get_shape())
            print(pool2.get_shape())

        with tf.variable_scope('deconv3'):

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                                   initializer=tf.constant_initializer(0.0))

            deconv3 = tf.nn.conv3d_transpose(pool2, filter=kernel,
                                             output_shape=[batch_size, 12, 12, 12, 10],
                                             strides=[1, 2, 2, 2, 1], padding='SAME')

            relu3 = tf.nn.relu(deconv3 + bias)
            drop3 = tf.nn.dropout(relu3, self.dropout_placeholder)

            print(deconv3.get_shape())
            print(relu3.get_shape())

        with tf.variable_scope('deconv4'):

            kernel = tf.get_variable('weights', [5, 5, 5, self.nb_classes, 10],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                                   initializer=tf.constant_initializer(0.0))

            deconv4 = tf.nn.conv3d_transpose(drop3, filter=kernel,
                                             output_shape=[batch_size, 24, 24, 24, self.nb_classes],
                                             strides=[1, 2, 2, 2, 1], padding='SAME')
            relu4 = tf.nn.relu(deconv4 + bias)

            print(deconv4.get_shape())
            print(relu4.get_shape())

        self.score = relu4

    def add_pred_op(self):
        probs = tf.nn.softmax(tf.reshape(self.score, [-1, self.nb_classes]))
        reshape_probs = tf.reshape(probs, tf.shape(self.score))

        self.pred = tf.argmax(reshape_probs, 4)
        self.prob = reshape_probs

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, self.nb_classes])
        labels = tf.reshape(self.label_placeholder, [-1])
        print(logits.shape)
        print(labels.shape)
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels))

        with tf.variable_scope('conv1', reuse=True):
            w1 = tf.get_variable('weights')
        with tf.variable_scope('conv2', reuse=True):
            w2 = tf.get_variable('weights')
        with tf.variable_scope('deconv3', reuse=True):
            w3 = tf.get_variable('weights')
        with tf.variable_scope('deconv4', reuse=True):
            w4 = tf.get_variable('weights')
        reg_loss = self.config.l2 * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
                                     + tf.nn.l2_loss(w3)
                                     + tf.nn.l2_loss(w4))

        self.loss = ce_loss + reg_loss

        # print(self.loss.get_shape())

######### use dice score as loss function
        # preds = tf.reshape(self.pred, [-1])
        # dice_score_loss = dice_score(labels.eval(), preds.eval())

# ######### l2 regularization for CNN model
#         with tf.variable_scope('conv1', reuse=True) as scope:
#             w1 = tf.get_variable('weights')
#         with tf.variable_scope('conv2', reuse=True) as scope:
#             w2 = tf.get_variable('weights')
#         with tf.variable_scope('conv3', reuse=True) as scope:
#             w3 = tf.get_variable('weights')
#         with tf.variable_scope('conv4', reuse=True) as scope:
#             w4 = tf.get_variable('weights')
#         reg_loss = self.config.l2 * (tf.nn.l2_loss(w1)
#                                    + tf.nn.l2_loss(w2)
#                                    + tf.nn.l2_loss(w3)
#                                    + tf.nn.l2_loss(w4)

        # self.loss = ce_loss + reg_loss
        # self.loss = dice_score_loss

    def get_variables_to_restore(self, level=3):
        var_names_to_restore = []
        if level > 1:
            var_names_to_restore += ['conv1/weights',
                                     'conv1/biases']
        if level > 2:
            var_names_to_restore += ['conv2/weights',
                                     'conv2/biases']

        var_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=var_names_to_restore)
        var_to_train = tf.contrib.framework.get_variables_to_restore(include=var_names_to_restore)
        return var_to_train, var_to_restore

    def add_train_op(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                 .minimize(self.loss)

            if self.config.finetuning_method == 'last_layers':
                var_to_train, _ = self.get_variables_to_restore(self.config.finetuning_level)
                self.train_last_layers = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                                 .minimize(self.loss, var_list=var_to_train)

    def _train(self, ex_path, sess, lr, finetune=False):
        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches

        for _, (x, y) in enumerate(fcn_data_iter_v2(ex_path, 'fgbg', bs, nb, self.patch)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: self.config.dropout,
                    self.lr_placeholder: lr,
                    self.is_training: self.config.use_batch_norm}
            if finetune:
                pred, loss, _ = sess.run([self.pred, self.loss, self.train_last_layers], feed_dict=feed)
            else:
                pred, loss, _ = sess.run([self.pred, self.loss, self.train], feed_dict=feed)

            losses.append(loss)

            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return losses, bdices

    def _validate(self, ex_path, sess):
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_val_batches

        for _, (x, y) in enumerate(fcn_data_iter_v2(ex_path, 'fgbg', bs, nb, self.patch)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0,
                    self.is_training: False}

            pred = sess.run(self.pred, feed_dict=feed)
            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return bdices

    def _segment(self, ex_path, sess):
        fpred = np.zeros(get_shape_v2(ex_path))
        fy = np.zeros(get_shape_v2(ex_path))
        fprob = np.zeros(get_shape_v2(ex_path) + (2,))

        bs = self.config.batch_size

        for _, (i, j, k, x, y) in enumerate(fcn_data_iter_v2(ex_path, 'full', bs, None, self.patch)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0,
                    self.is_training:False}

            pred, prob = sess.run([self.pred, self.prob], feed_dict=feed)

            for idx, _ in enumerate(i):
                fy[i[idx]:i[idx] + self.patch,
                   j[idx]:j[idx] + self.patch,
                   k[idx]:k[idx] + self.patch] = y[idx, :, :, :]
                fpred[i[idx]:i[idx] + self.patch,
                      j[idx]:j[idx] + self.patch,
                      k[idx]:k[idx] + self.patch] = pred[idx, :, :, :]
                fprob[i[idx]:i[idx] + self.patch,
                      j[idx]:j[idx] + self.patch,
                      k[idx]:k[idx] + self.patch, :] = prob[idx, :, :, :, :]

        fdice = dice_score(fy, fpred)

        return fy, fpred, fprob, fdice
