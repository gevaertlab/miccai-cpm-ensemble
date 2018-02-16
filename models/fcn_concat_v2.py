import tensorflow as tf

from models.fcn_concat import FCN_Concat

class FCN_Concat_v2(FCN_Concat):
    def add_model(self):
        self.image = tf.reshape(self.image, [-1, self.patch, self.patch, self.patch, self.nb_modalities])
        nb_filters = self.config.nb_filters
        k_size = self.config.kernel_size

        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(inputs=self.image,
                                     filters=nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)
            bn1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)
            relu1 = tf.nn.relu(bn1)

            conv1 = tf.layers.conv3d(inputs=relu1,
                                     filters=nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)
            bn1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)
            relu1 = tf.nn.relu(bn1)

            # shape = (patch/2, patch/2, patch/2)
            pool1 = tf.layers.max_pooling3d(inputs=relu1, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            # drop1 = tf.nn.dropout(pool1, self.dropout_placeholder)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv3d(inputs=pool1,
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training)
            relu2 = tf.nn.relu(bn2)

            conv2 = tf.layers.conv3d(inputs=relu2,
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training)
            relu2 = tf.nn.relu(bn2)

            # shape = (patch/4, patch/4, patch/4)
            pool2 = tf.layers.max_pooling3d(inputs=relu2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            # drop2 = tf.nn.dropout(pool2, self.dropout_placeholder)

        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv3d(inputs=pool2,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.is_training)
            relu3 = tf.nn.relu(bn3)

            conv3 = tf.layers.conv3d(inputs=relu3,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.is_training)
            relu3 = tf.nn.relu(bn3)

            # shape = (patch/8, patch/8, patch/8)
            pool3 = tf.layers.max_pooling3d(inputs=relu3, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop3 = tf.nn.dropout(pool3, self.dropout_placeholder)

            # print(conv.get_shape())

        with tf.variable_scope('deconv4'):
            deconv4 = tf.layers.conv3d_transpose(inputs=drop3,
                                                 filters=2 * nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 kernel_regularizer=tf.nn.l2_loss)

            bias = tf.get_variable('biases', [2 * nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv4 = deconv4 + bias
            # shape = (patch/4, patch/4, patch/4)
            bn4 = tf.layers.batch_normalization(deconv4, axis=-1, training=self.is_training)
            bn4 = tf.concat([bn4, pool2], axis=-1)
            relu4 = tf.nn.relu(bn4)
            drop4 = tf.nn.dropout(relu4, self.dropout_placeholder)

            conv4 = tf.layers.conv3d(inputs=drop4,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn4 = tf.layers.batch_normalization(conv4, axis=-1, training=self.is_training)
            relu4 = tf.nn.relu(bn4)
            drop4 = tf.nn.dropout(relu4, self.dropout_placeholder)

        with tf.variable_scope('deconv5'):
            deconv5 = tf.layers.conv3d_transpose(inputs=drop4,
                                                 filters=nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0),
                                                 kernel_regularizer=tf.nn.l2_loss)
            bias = tf.get_variable('biases', [nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv5 = deconv5 + bias
            # shape = (patch/2, patch/2, patch/2)
            bn5 = tf.layers.batch_normalization(deconv5, axis=-1, training=self.is_training)
            bn5 = tf.concat([bn5, pool1], axis=-1)
            relu5 = tf.nn.relu(bn5)

            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

            conv5 = tf.layers.conv3d(inputs=drop5,
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     kernel_regularizer=tf.nn.l2_loss)

            bn5 = tf.layers.batch_normalization(conv5, axis=-1, training=self.is_training)
            relu5 = tf.nn.relu(bn5)
            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

            # print(deconv.get_shape())
            # print(relu5.get_shape())

        with tf.variable_scope('deconv6'):
            deconv6 = tf.layers.conv3d_transpose(inputs=drop5,
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
            bias = tf.get_variable('biases', [self.nb_classes],
                                   initializer=tf.zeros_initializer())
            self.score = deconv6 + bias