""""
USELESS !!!!!!!!!!!!!!!!!!!!
REMOVE THIS FILE
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.fcn import FCNBaselineModel

class FCN_Model(FCNBaselineModel):

    def add_model(self):

        with tf.variable_scope('conv1') as scope:

            conv1 = tf.layers.conv3d(inputs=self.image_placeholder, filters=10, kernel_size=[5, 5, 5], padding="SAME", use_bias=True,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
            
            pool1 = tf.layers.max_pooling3d(inputs = conv1, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')

            print(conv1.get_shape())
            print(pool1.get_shape())

        with tf.variable_scope('conv2') as scope:

            conv2 = tf.layers.conv3d(inputs=pool1, filters=20, kernel_size=[5, 5, 5], padding="SAME", use_bias=True,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
            
            pool2 = tf.layers.max_pooling3d(inputs = conv2, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')

            print(conv2.get_shape())
            print(pool2.get_shape())


        with tf.variable_scope('deconv2') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                    initializer=tf.constant_initializer(0.0))

            deconv3 = tf.nn.conv3d_transpose(pool2, filter = kernel, output_shape = [-1, 12, 12, 12, 10], strides = [1,2,2,2,1], padding='VALID')
            relu3 = tf.nn.relu(deconv3 + bias)

            print(deconv3.get_shape())
            print(relu3.get_shape())

        with tf.variable_scope('deconv3') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 2, 10],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                    initializer=tf.constant_initializer(0.0))

            deconv4 = tf.nn.conv3d_transpose(relu3, filter = kernel, output_shape = [-1, 24, 24, 24, 2], strides = [1,1,1,1,1], padding='VALID')
            relu4 = tf.nn.relu(deconv4 + bias)

            print(deconv4.get_shape())
            print(relu4.get_shape())

        self.score = relu4



def DeeperFCN()
        batch_size = tf.shape(self.label_placeholder)[0]
        with tf.variable_scope('conv1') as scope:

            conv1 = tf.layers.conv3d(inputs=self.image_placeholder, filters=10, kernel_size=[5, 5, 5], padding="SAME", use_bias=True,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
            drop1 = tf.nn.dropout(conv1, self.dropout_placeholder)
            pool1 = tf.layers.max_pooling3d(inputs = drop1, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')

            print(conv1.get_shape())
            print(pool1.get_shape())

        with tf.variable_scope('conv2') as scope:

            conv2 = tf.layers.conv3d(inputs=pool1, filters=20, kernel_size=[5, 5, 5], padding="SAME", use_bias=True,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
            drop2 = tf.nn.dropout(conv2, self.dropout_placeholder)
            pool2 = tf.layers.max_pooling3d(inputs = drop2, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')

            print(conv2.get_shape())
            print(pool2.get_shape())


        with tf.variable_scope('conv3') as scope:

            conv3 = tf.layers.conv3d(inputs=pool2, filters=40, kernel_size=[5, 5, 5], padding="SAME", use_bias=True,
                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
            drop3 = tf.nn.dropout(conv3, self.dropout_placeholder)
            pool3 = tf.layers.max_pooling3d(inputs = drop3, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')

            print(conv3.get_shape())
            print(pool3.get_shape())


        with tf.variable_scope('deconv4') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 20, 40],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                    initializer=tf.constant_initializer(0.0))

            deconv4 = tf.nn.conv3d_transpose(pool3, filter = kernel, output_shape = [batch_size, 6, 6, 6, 20], strides = [1,2,2,2,1], padding='SAME')
            relu4 = tf.nn.relu(deconv4 + bias)
            drop4 = tf.nn.dropout(relu4, self.dropout_placeholder)

            print(deconv4.get_shape())
            print(relu4.get_shape())
            print(drop4.get_shape())


        with tf.variable_scope('deconv5') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                    initializer=tf.constant_initializer(0.0))

            deconv5 = tf.nn.conv3d_transpose(drop4, filter = kernel, output_shape = [batch_size, 12, 12, 12, 10], strides = [1,2,2,2,1], padding='SAME')
            relu5 = tf.nn.relu(deconv5 + bias)
            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

            print(deconv5.get_shape())
            print(relu5.get_shape())
            print(drop5.get_shape())

        with tf.variable_scope('deconv6') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 2, 10],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                    initializer=tf.constant_initializer(0.0))

            deconv6 = tf.nn.conv3d_transpose(drop5, filter = kernel, output_shape = [batch_size, 24, 24, 24, 2], strides = [1,2,2,2,1], padding='SAME')
            relu6 = tf.nn.relu(deconv6 + bias)

            print(deconv6.get_shape())
            print(relu6.get_shape())
            

        self.score = relu6


################## 3D deconv layers

        # with tf.variable_scope('deconv2') as scope:

        #     deconv2 = tf.layers.conv3d_transpose(pool2, filters=10, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding="SAME", use_bias=True,
        #         activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))


        # with tf.variable_scope('deconv1') as scope:

        #     deconv2 = tf.layers.conv3d_transpose(deconv1, filters=2, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding="SAME", use_bias=True,
        #         activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.0))
           
        # self.score = deconv2


################# CNN model
        # with tf.variable_scope('conv1') as scope:
        #     kernel = tf.get_variable('weights', [5, 5, 5, 4, 10],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [10],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(self.image_placeholder, kernel,
        #                         [1, 1, 1, 1, 1], padding='VALID')
        #     relu1 = tf.nn.relu(conv + bias)

        # with tf.variable_scope('conv2') as scope:
        #     kernel = tf.get_variable('weights', [5, 5, 5, 10, 15],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [15],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(relu1, kernel,
        #                         [1, 1, 1, 1, 1], padding='VALID')
        #     relu2 = tf.nn.relu(conv + bias)

        # with tf.variable_scope('conv3') as scope:
        #     kernel = tf.get_variable('weights', [5, 5, 5, 15, 15],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [15],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(relu2, kernel,
        #                         [1, 1, 1, 1, 1], padding='VALID')
        #     relu3 = tf.nn.relu(conv + bias)

        # with tf.variable_scope('conv4') as scope:
        #     kernel = tf.get_variable('weights', [5, 5, 5, 15, 20],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [20],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(relu3, kernel,
        #                         [1, 1, 1, 1, 1], padding='VALID')
        #     relu4 = tf.nn.relu(conv + bias)

        # with tf.variable_scope('fc1') as scope:
        #     kernel = tf.get_variable('weights', [1, 1, 1, 20, 80],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [80],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(relu4, kernel,
        #                         [1, 1, 1, 1, 1], padding='SAME')
        #     relu = tf.nn.relu(conv + bias)
        #     drop5 = tf.nn.dropout(relu, self.dropout_placeholder)

        # with tf.variable_scope('fc2') as scope:
        #     kernel = tf.get_variable('weights', [1, 1, 1, 80, 2],
        #              initializer=tf.contrib.layers.xavier_initializer())
        #     bias = tf.get_variable('biases', [2],
        #            initializer=tf.constant_initializer(0.0))

        #     conv = tf.nn.conv3d(drop5, kernel,
        #                         [1, 1, 1, 1, 1], padding='SAME')
        #     self.score = conv + bias

