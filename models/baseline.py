import numpy as np
import tensorflow as tf

from models.model import Model

from utils.data_iterator import data_iter
from utils.data_utils import get_ex_paths, get_shape
from utils.dice_score import dice_score


class BaselineModel(Model):

    def __init__(self, config):
        self.config = config

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
        self.image_placeholder = tf.placeholder(tf.float32, shape=[None, 25, 25, 25, 4])
        self.label_placeholder = tf.placeholder(tf.int32, shape=[None, 9, 9, 9])
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[])
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])

    def add_model(self):

        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[5, 5, 5, 4, 30],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[30],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(self.image_placeholder, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu1 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[5, 5, 5, 30, 40],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[40],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu1, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu2 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[5, 5, 5, 40, 40],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[40],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu2, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu3 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[5, 5, 5, 40, 50],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[50],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu3, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu4 = tf.nn.relu(conv + bias)

        with tf.variable_scope('fc1') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[1, 1, 1, 50, 100],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[100],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu4, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)
            drop5 = tf.nn.dropout(relu, self.dropout_placeholder)

        with tf.variable_scope('fc2') as scope:
            kernel = tf.get_variable('weights',
                                     shape=[1, 1, 1, 100, 2],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases',
                                   shape=[2],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop5, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            self.score = conv + bias

    def add_pred_op(self):
        probs = tf.nn.softmax(tf.reshape(self.score, [-1, 2]))
        reshape_probs = tf.reshape(probs, tf.shape(self.score))

        self.pred = tf.argmax(reshape_probs, 4)
        self.prob = reshape_probs

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, 2])
        labels = tf.reshape(self.label_placeholder, [-1])
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels))

######### use dice score as loss function
        # preds = tf.reshape(self.pred, [-1])
        # dice_score_loss = dice_score(labels.eval(), preds.eval())
        # self.loss = dice_score_loss

######### l2 regularization for CNN model
        with tf.variable_scope('conv1', reuse=True) as scope:
            w1 = tf.get_variable('weights')
        with tf.variable_scope('conv2', reuse=True) as scope:
            w2 = tf.get_variable('weights')
        with tf.variable_scope('conv3', reuse=True) as scope:
            w3 = tf.get_variable('weights')
        with tf.variable_scope('conv4', reuse=True) as scope:
            w4 = tf.get_variable('weights')
        with tf.variable_scope('fc1', reuse=True) as scope:
            wfc1 = tf.get_variable('weights')
        with tf.variable_scope('fc2', reuse=True) as scope:
            wfc2 = tf.get_variable('weights')
        reg_loss = self.config.l2 * (tf.nn.l2_loss(w1)
                                     + tf.nn.l2_loss(w2)
                                     + tf.nn.l2_loss(w3)
                                     + tf.nn.l2_loss(w4)
                                     + tf.nn.l2_loss(wfc1)
                                     + tf.nn.l2_loss(wfc2))

        self.loss = ce_loss + reg_loss

    def add_train_op(self):
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder).minimize(self.loss)

    def _train(self, ex_path, sess, lr):

        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches

        for _, (x, y) in enumerate(data_iter(ex_path, 'fgbg', bs, nb)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: self.config.dropout,
                    self.lr_placeholder: lr}

            pred, loss, _ = sess.run([self.pred, self.loss, self.train], feed_dict=feed)

            losses.append(loss)

            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return losses, bdices

    def _validate(self, ex_path, sess):

        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_val_batches

        for _, (x, y) in enumerate(data_iter(ex_path, 'fgbg', bs, nb)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred = sess.run(self.pred, feed_dict=feed)
            # print(type(y), type(pred))
            # sys.exit()
            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return bdices

    def _segment(self, ex_path, sess):

        fpred = np.zeros(get_shape(ex_path))
        fy = np.zeros(get_shape(ex_path))
        fprob = np.zeros(get_shape(ex_path) + (2,))

        bs = self.config.batch_size

        for _, (i, j, k, x, y) in enumerate(data_iter(ex_path, 'full', bs, None)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred, prob = sess.run([self.pred, self.prob], feed_dict=feed)

            for idx, _ in enumerate(i):
                fy[i[idx] - 4:i[idx] + 5,
                   j[idx] - 4:j[idx] + 5,
                   k[idx] - 4:k[idx] + 5] = y[idx, :, :, :]
                fpred[i[idx] - 4:i[idx] + 5,
                      j[idx] - 4:j[idx] + 5,
                      k[idx] - 4:k[idx] + 5] = pred[idx, :, :, :]
                fprob[i[idx] - 4:i[idx] + 5,
                      j[idx] - 4:j[idx] + 5,
                      k[idx] - 4:k[idx] + 5, :] = prob[idx, :, :, :, :]

        fdice = dice_score(fy, fpred)

        return fy, fpred, fprob, fdice
