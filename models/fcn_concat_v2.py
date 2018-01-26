import tensorflow as tf

from models.fcn_concat import FCN_Concat

class FCN_Concat_v2(FCN_Concat):
    def add_model(self):
        self.image = tf.reshape(self.image, [-1, self.patch, self.patch, self.patch, 4])
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

            # shape = (patch/8, patch/8, patch/8)
            pool3 = tf.layers.max_pooling3d(inputs=relu3, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop3 = tf.nn.dropout(pool3, self.dropout_placeholder)

            # print(conv.get_shape())

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv3d(inputs=drop3,
                                     filters=8 * nb_filters,
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

            pool4 = tf.layers.max_pooling3d(inputs=relu4, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop4 = tf.nn.dropout(pool4, self.dropout_placeholder)

        with tf.variable_scope('deconv5'):
            deconv5 = tf.layers.conv3d_transpose(inputs=drop4,
                                                 filters=4 * nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 kernel_regularizer=tf.nn.l2_loss)

            bias = tf.get_variable('biases', [4 * nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv5 = deconv5 + bias
            bn5 = tf.layers.batch_normalization(deconv5, axis=-1, training=self.is_training)
            bn5 = tf.concat([bn5, pool3], axis=-1)
            relu5 = tf.nn.relu(bn5)

            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

        with tf.variable_scope('deconv6'):
            deconv6 = tf.layers.conv3d_transpose(inputs=drop5,
                                                 filters=2 * nb_filters,
                                                 kernel_size=k_size,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0),
                                                 kernel_regularizer=tf.nn.l2_loss)
            bias = tf.get_variable('biases', [2 * nb_filters],
                                   initializer=tf.zeros_initializer())
            deconv6 = deconv6 + bias
            # shape = (patch/2, patch/2, patch/2)
            bn6 = tf.layers.batch_normalization(deconv6, axis=-1, training=self.is_training)
            bn6 = tf.concat([bn6, pool2], axis=-1)
            relu6 = tf.nn.relu(bn6)

            drop6 = tf.nn.dropout(relu6, self.dropout_placeholder)

            # print(deconv.get_shape())
            # print(relu5.get_shape())

        with tf.variable_scope('deconv7'):
            deconv7 = tf.layers.conv3d_transpose(inputs=drop6,
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
            deconv7 = deconv7 + bias
            # shape = (patch/2, patch/2, patch/2)
            bn7 = tf.layers.batch_normalization(deconv7, axis=-1, training=self.is_training)
            bn7 = tf.concat([bn7, pool1], axis=-1)
            relu7 = tf.nn.relu(bn7)

            drop7 = tf.nn.dropout(relu7, self.dropout_placeholder)

        with tf.variable_scope('deconv8'):
            deconv8 = tf.layers.conv3d_transpose(inputs=drop7,
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
            self.score = deconv8 + bias

    def add_loss_op(self):
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.label)

        # add mask
        if self.config.use_mask:
            mask = tf.get_variable('mask', shape=(self.patch, self.patch, self.patch),
                                   dtype=tf.int32, initializer=tf.zeros_initializer())
            c_size = self.config.center_patch
            lower = self.patch // 2 - c_size // 2
            center = tf.ones(shape=(c_size, c_size, c_size), dtype=tf.int32)
            mask = tf.assign(mask[lower: lower + c_size, lower: lower + c_size, lower: lower + c_size], center)
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=0)
            mask = tf.tile(mask, multiples=[tf.shape(ce_loss)[0], 1, 1, 1])
            ce_loss = tf.boolean_mask(ce_loss, mask)

        ce_loss = tf.reduce_mean(ce_loss)
        reg_loss = self.config.l2 * tf.losses.get_regularization_loss()
        self.loss = ce_loss + reg_loss
