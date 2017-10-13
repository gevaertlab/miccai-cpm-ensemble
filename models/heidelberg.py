from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.baseline import BaselineModel

from utils.data_iterator import heidelberg_iter
from utils.data_utils import get_shape_hb

from sklearn.metrics import f1_score

class HeidelbergModel(BaselineModel):

    def add_placeholders(self):
        self.image_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, 25, 25, 25, 5])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                shape=[None, 9, 9, 9])
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  shape=[])

    def add_model(self):

        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 5, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(self.image_placeholder, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu1 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 10, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu1, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu2 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu2, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu3 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu3, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu4 = tf.nn.relu(conv + bias)

        with tf.variable_scope('fc1') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 20, 80],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [80],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu4, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)
            drop5 = tf.nn.dropout(relu, self.dropout_placeholder)

        with tf.variable_scope('fc2') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 80, 4],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [4],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop5, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            self.score = conv + bias

    def add_pred_op(self):
        probs = tf.nn.softmax(tf.reshape(self.score, [-1, 4]))
        reshape_probs = tf.reshape(probs, tf.shape(self.score))

        self.pred = tf.argmax(reshape_probs, 4)
        self.prob = reshape_probs

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, 4])
        labels = tf.reshape(self.label_placeholder, [-1])
        ce_loss = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits, labels))
        
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

    def _train(self, ex_path, sess):
        
        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches
        
        for batch, (x, y) in enumerate(heidelberg_iter(
                                       ex_path, 'fgbg', bs, nb)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 0.5}

            pred, loss, _ = sess.run([self.pred, self.loss, self.train],
                            feed_dict=feed)

            losses.append(loss)
            bdice = f1_score(np.ravel(y), np.ravel(pred),
                             labels=[0, 1, 2, 3], average='macro')
            bdices.append(bdice)
            
        return losses, bdices

    def _validate(self, ex_path, sess):
        
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_val_batches

        for batch, (x, y) in enumerate(heidelberg_iter(
                                       ex_path, 'fgbg', bs, nb)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred = sess.run(self.pred, feed_dict=feed)

            bdice = f1_score(np.ravel(y), np.ravel(pred),
                             labels=[0, 1, 2, 3], average='macro')
            bdices.append(bdice)

        return bdices

    def _segment(self, ex_path, sess):
        
        fpred = np.zeros(get_shape_hb(ex_path))
        fy = np.zeros(get_shape_hb(ex_path))
        fprob = np.zeros(get_shape_hb(ex_path) + (4,))

        bs = self.config.batch_size

        for batch, (i, j, k, x, y) in enumerate(heidelberg_iter(
                                                ex_path, 'full', bs, None)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred, prob = sess.run([self.pred, self.prob], feed_dict=feed)

            for idx, _ in enumerate(i):
                fy[i[idx]-4:i[idx]+5,
                   j[idx]-4:j[idx]+5,
                   k[idx]-4:k[idx]+5] = y[idx, :, :, :]
                fpred[i[idx]-4:i[idx]+5,
                      j[idx]-4:j[idx]+5,
                      k[idx]-4:k[idx]+5] = pred[idx, :, :, :]
                fprob[i[idx]-4:i[idx]+5,
                      j[idx]-4:j[idx]+5,
                      k[idx]-4:k[idx]+5, :] = prob[idx, :, :, :, :]
        
        fdice = f1_score(np.ravel(fy), np.ravel(fpred),
                         labels=[0, 1, 2, 3], average='macro')

        return fy, fpred, fprob, fdice
