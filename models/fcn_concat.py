import tensorflow as tf

from models.fcn import FCN_Model


class FCN_Concat(FCN_Model):

    def add_model(self):
        # TODO: try activations before batch norm

        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(inputs=self.image_placeholder,
                                     filters=10,
                                     kernel_size=5,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))

            bn1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)
            relu1 = tf.nn.relu(bn1)

            # shape = (patch/2, patch/2, patch/2)
            pool1 = tf.layers.max_pooling3d(inputs=relu1, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            # drop1 = tf.nn.dropout(pool1, self.dropout_placeholder)

            # print(conv1.get_shape())

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv3d(inputs=pool1,
                                     filters=20,
                                     kernel_size=5,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))

            bn2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training)
            relu2 = tf.nn.relu(bn2)

            # shape = (patch/4, patch/4, patch/4)
            pool2 = tf.layers.max_pooling3d(inputs=relu2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            # drop2 = tf.nn.dropout(pool2, self.dropout_placeholder)
            # print(conv.get_shape())

        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv3d(inputs=pool2,
                                     filters=40,
                                     kernel_size=5,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))

            bn3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.is_training)
            relu3 = tf.nn.relu(bn3)

            # shape = (patch/8, patch/8, patch/8)
            pool3 = tf.layers.max_pooling3d(inputs=relu3, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop3 = tf.nn.dropout(pool3, self.dropout_placeholder)

            # print(conv.get_shape())

        with tf.variable_scope('deconv4'):
            deconv4 = tf.layers.conv3d_transpose(inputs=drop3,
                                                 filters=20,
                                                 kernel_size=5,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.zeros_initializer())

            bias = tf.get_variable('biases', [20],
                                   initializer=tf.zeros_initializer())
            deconv4 = deconv4 + bias
            # shape = (patch/4, patch/4, patch/4)
            bn4 = tf.layers.batch_normalization(deconv4, axis=-1, training=self.is_training)
            bn4 = tf.concat([bn4, pool2], axis=-1)
            relu4 = tf.nn.relu(bn4)

            drop4 = tf.nn.dropout(relu4, self.dropout_placeholder)

        with tf.variable_scope('deconv5'):
            deconv5 = tf.layers.conv3d_transpose(inputs=drop4,
                                                 filters=10,
                                                 kernel_size=5,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))
            bias = tf.get_variable('biases', [10],
                                   initializer=tf.zeros_initializer())
            deconv5 = deconv5 + bias
            # shape = (patch/2, patch/2, patch/2)
            bn5 = tf.layers.batch_normalization(deconv5, axis=-1, training=self.is_training)
            bn5 = tf.concat([bn5, pool1], axis=-1)
            relu5 = tf.nn.relu(bn5)

            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

            # print(deconv.get_shape())
            # print(relu5.get_shape())

        with tf.variable_scope('deconv6'):
            deconv6 = tf.layers.conv3d_transpose(inputs=drop5,
                                                 filters=2,
                                                 kernel_size=5,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))

            # print(deconv6.get_shape())
            bias = tf.get_variable('biases', [2],
                                   initializer=tf.zeros_initializer()) 
            self.score = deconv6 + bias

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, 2])
        labels = tf.reshape(self.label_placeholder, [-1])

        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        
        with tf.variable_scope('conv1', reuse=True):
            w1 = tf.get_variable('conv3d/kernel')
        with tf.variable_scope('conv2', reuse=True):
            w2 = tf.get_variable('conv3d/kernel')
        with tf.variable_scope('conv3', reuse=True):
            w3 = tf.get_variable('conv3d/kernel')
        with tf.variable_scope('deconv4', reuse=True):
            w4 = tf.get_variable('conv3d_transpose/kernel')
        with tf.variable_scope('deconv5', reuse=True):
            w5 = tf.get_variable('conv3d_transpose/kernel')
        with tf.variable_scope('deconv6', reuse=True):
            w6 = tf.get_variable('conv3d_transpose/kernel')
        reg_loss = self.config.l2 * (tf.nn.l2_loss(w1)
                                     + tf.nn.l2_loss(w2)
                                     + tf.nn.l2_loss(w3)
                                     + tf.nn.l2_loss(w4)
                                     + tf.nn.l2_loss(w5)
                                     + tf.nn.l2_loss(w6))

        self.loss = ce_loss + reg_loss

    def get_variables_to_restore(self, level=4):
        var_names_to_restore = ['conv1/conv3d/kernel:0',
                                'conv1/conv3d/bias:0',
                                'conv2/conv3d/kernel:0',
                                'conv2/conv3d/bias:0',
                                'conv3/conv3d/kernel:0',
                                'conv3/conv3d/bias:0']
        if level > 1:
            var_names_to_restore += ['deconv4/conv3d_transpose/kernel:0',
                                     'deconv4/biases:0']
        if level > 2:
            var_names_to_restore += ['deconv5/conv3d_transpose/kernel:0',
                                     'deconv5/biases:0']
        if level > 3:
            var_names_to_restore += ['deconv6/conv3d_transpose/kernel:0',
                                     'deconv6/biases:0']

        var_to_restore = tf.contrib.framework.get_variables_to_restore(include=var_names_to_restore)
        var_to_train = tf.contrib.framework.get_variables_to_restore(exclude=var_names_to_restore)
        # print('*' * 20 + 'variables to retrain' + '*' * 50)
        # print([var.name for var in var_to_train])
        # print('*' * 20 + 'variables to restore' + '*' * 50)
        # print([var.name for var in var_to_restore])
        return var_to_train, var_to_restore
