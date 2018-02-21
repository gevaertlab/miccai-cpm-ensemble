import tensorflow as tf

from models.fcn_concat import FCN_Concat

class FCN_Concat_v2(FCN_Concat):
    def add_model(self):
        self.image = tf.reshape(self.image, [-1, self.patch, self.patch, self.patch, self.nb_modalities])
        nb_filters = self.config.nb_filters
        k_size = self.config.kernel_size

        with tf.variable_scope('conv1'):
            conv1_1 = tf.layers.conv3d(inputs=self.image,
                                     filters=nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)
            bn1_1 = tf.layers.batch_normalization(conv1_1, axis=-1, training=self.is_training)
            relu1_1 = tf.nn.relu(bn1_1)

            conv1_2 = tf.layers.conv3d(inputs=relu1_1,
                                     filters=nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)
            bn1_2 = tf.layers.batch_normalization(conv1_2, axis=-1, training=self.is_training)
            relu1_2 = tf.nn.relu(bn1_2)

            # shape = (patch/2, patch/2, patch/2)
            pool1_2 = tf.layers.max_pooling3d(inputs=relu1_2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            # drop1 = tf.nn.dropout(pool1_2, self.dropout_placeholder)

        with tf.variable_scope('conv2'):
            conv2_1 = tf.layers.conv3d(inputs=pool1_2,
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn2_1 = tf.layers.batch_normalization(conv2_1, axis=-1, training=self.is_training)
            relu2_1 = tf.nn.relu(bn2_1)

            conv2_2 = tf.layers.conv3d(inputs=relu2_1,
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn2_2 = tf.layers.batch_normalization(conv2_2, axis=-1, training=self.is_training)
            relu2_2 = tf.nn.relu(bn2_2)

            # shape = (patch/4, patch/4, patch/4)
            pool2_2 = tf.layers.max_pooling3d(inputs=relu2_2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            # drop2 = tf.nn.dropout(pool2, self.dropout_placeholder)

        with tf.variable_scope('conv3'):
            conv3_1 = tf.layers.conv3d(inputs=pool2_2,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn3_1 = tf.layers.batch_normalization(conv3_1, axis=-1, training=self.is_training)
            relu3_1 = tf.nn.relu(bn3_1)
            drop3_1 = tf.nn.dropout(relu3_1, self.dropout_placeholder)

            conv3_2 = tf.layers.conv3d(inputs=drop3_1,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn3_2 = tf.layers.batch_normalization(conv3_2, axis=-1, training=self.is_training)
            relu3_2 = tf.nn.relu(bn3_2)

            # shape = (patch/8, patch/8, patch/8)
            pool3_2 = tf.layers.max_pooling3d(inputs=relu3_2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop3_2 = tf.nn.dropout(pool3_2, self.dropout_placeholder)

            # print(conv.get_shape())

        with tf.variable_scope('deconv4'):
            deconv4_1 = tf.layers.conv3d_transpose(inputs=drop3_2,
                                                 filters=2 * nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 kernel_regularizer=tf.nn.l2_loss)

            bias_4 = tf.get_variable('biases', [2 * nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv4_1 = deconv4_1 + bias_4
            # shape = (patch/4, patch/4, patch/4)
            bn4_1 = tf.layers.batch_normalization(deconv4_1, axis=-1, training=self.is_training)
            bn4_1 = tf.concat([bn4_1, pool2_2], axis=-1)
            relu4_1 = tf.nn.relu(bn4_1)
            drop4_1 = tf.nn.dropout(relu4_1, self.dropout_placeholder)

            conv4_2 = tf.layers.conv3d(inputs=drop4_1,
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn4_2 = tf.layers.batch_normalization(conv4_2, axis=-1, training=self.is_training)
            relu4_2 = tf.nn.relu(bn4_2)
            drop4_2 = tf.nn.dropout(relu4_2, self.dropout_placeholder)

        with tf.variable_scope('deconv5'):
            deconv5_1 = tf.layers.conv3d_transpose(inputs=drop4_2,
                                                 filters=nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0),
                                                 kernel_regularizer=tf.nn.l2_loss)
            bias_5 = tf.get_variable('biases', [nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv5_1 = deconv5_1 + bias_5
            # shape = (patch/2, patch/2, patch/2)
            bn5_1 = tf.layers.batch_normalization(deconv5_1, axis=-1, training=self.is_training)
            bn5_1 = tf.concat([bn5_1, pool1_2], axis=-1)
            relu5_1 = tf.nn.relu(bn5_1)

            drop5_1 = tf.nn.dropout(relu5_1, self.dropout_placeholder)

            conv5_2 = tf.layers.conv3d(inputs=drop5_1,
                                     filters=nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn5_2 = tf.layers.batch_normalization(conv5_2, axis=-1, training=self.is_training)
            relu5_2 = tf.nn.relu(bn5_2)
            drop5_2 = tf.nn.dropout(relu5_2, self.dropout_placeholder)

            # print(deconv.get_shape())
            # print(relu5.get_shape())

        with tf.variable_scope('deconv6'):
            deconv6_1 = tf.layers.conv3d_transpose(inputs=drop5_2,
                                                 filters=self.nb_classes,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0),
                                                 kernel_regularizer=tf.nn.l2_loss)

            # print(deconv6.get_shape())
            bias_6 = tf.get_variable('biases', [self.nb_classes],
                                   initializer=tf.zeros_initializer())

            bn6_1 = tf.layers.batch_normalization(deconv5_1, axis=-1, training=self.is_training)
            relu6_1 = tf.nn.relu(bn6_1)
            drop6_1 = tf.nn.dropout(relu6_1, self.dropout_placeholder)

            conv6_2 = tf.layers.conv3d(inputs=drop6_1,
                                     filters=self.nb_classes,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            self.score = conv6_2