import numpy as np
import tensorflow as tf

from models.baseline import BaselineModel

class BaselineDropoutModel(BaselineModel):

    def add_model(self):
        keep_rate = 0.8
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 4, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(self.image_placeholder, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu1 = tf.nn.relu(conv + bias)
            drop1 = tf.nn.dropout(relu1, keep_prob=keep_rate)

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 10, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop1, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu2 = tf.nn.relu(conv + bias)
            drop2 = tf.nn.dropout(relu2, keep_prob=keep_rate)

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [15],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop2, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu3 = tf.nn.relu(conv + bias)
            drop3 = tf.nn.dropout(relu3, keep_prob=keep_rate)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', [5, 5, 5, 15, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop3, kernel,
                                [1, 1, 1, 1, 1], padding='VALID')
            relu4 = tf.nn.relu(conv + bias)
            drop4 = tf.nn.dropout(relu4, keep_prob=keep_rate)

        with tf.variable_scope('fc1') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 20, 80],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [80],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop4, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            relu5 = tf.nn.relu(conv + bias)
            drop5 = tf.nn.dropout(relu5, keep_prob=self.dropout_placeholder)

        with tf.variable_scope('fc2') as scope:
            kernel = tf.get_variable('weights', [1, 1, 1, 80, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(drop5, kernel,
                                [1, 1, 1, 1, 1], padding='SAME')
            self.score = conv + bias
