from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.baseline import BaselineModel

from utils.data_iterator import data_iter
from utils.data_utils import get_ex_paths, get_shape
from utils.dice_score import dice_score

from skimage.measure import label
from skimage.morphology import binary_erosion, binary_dilation, ball

class NoT2Model(BaselineModel):

    def add_placeholders(self):
        self.image_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, 25, 25, 25, 2])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                shape=[None, 9, 9, 9])
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  shape=[])

    def add_model(self):

        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 2, 10],
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
            bias = tf.get_variable('biases', [40],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu2, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu3 = tf.nn.relu(conv + bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 40, 50],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu3, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu4 = tf.nn.relu(conv + bias)

        with tf.variable_scope('fc1') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 15, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(relu4, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)
            drop5 = tf.nn.dropout(relu, self.dropout_placeholder)

        with tf.variable_scope('fc2') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 80, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop5, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            self.score = conv + bias

    def _train(self, ex_path, sess):
        
        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches
        
	    for batch, (x, y) in enumerate(data_iter(
                                       ex_path, 'fgbg', bs, nb)):

            y = np.concatenate([y[..., 0], y[..., 3]], axis=4)

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 0.5}

            pred, loss, _ = sess.run([self.pred, self.loss, self.train],
                            feed_dict=feed)

            losses.append(loss)
            bdice = dice_score(y, pred)
            bdices.append(bdice)
            
        return losses, bdices

    def _validate(self, ex_path, sess):
        
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_val_batches

        for batch, (x, y) in enumerate(data_iter(
                                       ex_path, 'fgbg', bs, nb)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            y = np.concatenate([y[..., 0], y[..., 3]], axis=4)

            pred = sess.run(self.pred, feed_dict=feed)

            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return bdices

    def _segment(self, ex_path, sess):
        
        fpred = np.zeros(get_shape(ex_path))
        fy = np.zeros(get_shape(ex_path))

        bs = self.config.batch_size

        for batch, (i, j, k, x, y) in enumerate(data_iter(
                                                ex_path, 'full', bs, None)):

            y = np.concatenate([y[..., 0], y[..., 3]], axis=4)

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
        
        fdice = dice_score(fy, fpred)

        return fy, fpred, fprob, fdice
