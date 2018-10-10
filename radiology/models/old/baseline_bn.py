import numpy as np
import tensorflow as tf

from models.baseline import BaselineModel

class BaselineBnModel(BaselineModel):

    def add_model(self):

        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 4, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(self.image_placeholder, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            bn1 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu1 = tf.nn.relu(bn1)
            drop1 = tf.nn.dropout(relu1, keep_prob=self.dropout_placeholder)  # shape = (21, 21, 21)

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 10, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop1, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            bn2 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu2 = tf.nn.relu(bn2)
            drop2 = tf.nn.dropout(relu2, keep_prob=self.dropout_placeholder)  # shape = (17, 17, 17)

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop2, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            bn3 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu3 = tf.nn.relu(bn3)
            drop3 = tf.nn.dropout(relu3, keep_prob=self.dropout_placeholder)  # shape = (13, 13, 13)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop3, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            bn4 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu4 = tf.nn.relu(bn4)
            drop4 = tf.nn.dropout(relu4, keep_prob=self.dropout_placeholder)  # shape = (9, 9, 9)

        with tf.variable_scope('fc1') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 20, 80],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [80],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop4, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            bn5 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu5 = tf.nn.relu(bn5)
            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

        with tf.variable_scope('fc2') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 80, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop5, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            self.score = conv + bias
