import tensorflow as tf

from models.fcn import FCN_Model


class FCN_6_Model(FCN_Model):

    def add_model(self):
        batch_size = tf.shape(self.label_placeholder)[0]

        with tf.variable_scope('conv1') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 4, 10],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(input=self.image_placeholder, filter=kernel,
                                strides=[1, 1, 1, 1, 1], padding="SAME")

            bn1 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu1 = tf.nn.relu(bn1)

            pool1 = tf.layers.max_pooling3d(inputs=relu1, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')

            # drop1 = tf.nn.dropout(pool1, self.dropout_placeholder)

            # print(conv1.get_shape())
            print(pool1.get_shape())

        with tf.variable_scope('conv2') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(input=pool1, filter=kernel,
                                strides=[1, 1, 1, 1, 1], padding="SAME")

            bn2 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling3d(inputs=relu2, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            # drop2 = tf.nn.dropout(pool2, self.dropout_placeholder)
            # print(conv.get_shape())
            # print(pool2.get_shape())

        with tf.variable_scope('conv3') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 20, 40],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [40],
                                   initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(input=pool2, filter=kernel,
                                strides=[1, 1, 1, 1, 1], padding="SAME")

            bn3 = tf.layers.batch_normalization(conv + bias, axis=-1)
            relu3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling3d(inputs=relu3, pool_size=(2, 2, 2),
                                            strides=(2, 2, 2), padding='VALID')
            drop3 = tf.nn.dropout(pool3, self.dropout_placeholder)

            # print(conv.get_shape())
            # print(pool3.get_shape())

        with tf.variable_scope('deconv4') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 20, 40],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                                   initializer=tf.constant_initializer(0.0))

            out_dim = 2 * self.patch // (2 * 2 * 2)
            deconv = tf.nn.conv3d_transpose(drop3, filter=kernel,
                                            output_shape=[batch_size, out_dim, out_dim, out_dim, 20],
                                            strides=[1, 2, 2, 2, 1], padding='SAME')

            bn4 = tf.layers.batch_normalization(deconv + bias, axis=-1)
            relu4 = tf.nn.relu(bn4)

            drop4 = tf.nn.dropout(relu4, self.dropout_placeholder)

        with tf.variable_scope('deconv5') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                                   initializer=tf.constant_initializer(0.0))

            out_dim = 2 * self.patch // (2 * 2)
            deconv = tf.nn.conv3d_transpose(drop4, filter=kernel,
                                            output_shape=[batch_size, out_dim, out_dim, out_dim, 10],
                                            strides=[1, 2, 2, 2, 1], padding='SAME')

            bn5 = tf.layers.batch_normalization(deconv + bias, axis=-1)
            relu5 = tf.nn.relu(bn5)

            drop5 = tf.nn.dropout(relu5, self.dropout_placeholder)

            # print(deconv.get_shape())
            # print(relu5.get_shape())

        with tf.variable_scope('deconv6') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 2, 10],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                                   initializer=tf.constant_initializer(0.0))

            out_dim = 2 * self.patch // 2
            deconv6 = tf.nn.conv3d_transpose(drop5, filter=kernel,
                                             output_shape=[batch_size, out_dim, out_dim, out_dim, 2],
                                             strides=[1, 2, 2, 2, 1], padding='SAME')

            # print(deconv6.get_shape())

            self.score = deconv6 + bias

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, 2])
        labels = tf.reshape(self.label_placeholder, [-1])
        print(logits.shape)
        print(labels.shape)
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        with tf.variable_scope('conv1', reuse=True) as scope:
            w1 = tf.get_variable('weights')
        with tf.variable_scope('conv2', reuse=True) as scope:
            w2 = tf.get_variable('weights')
        with tf.variable_scope('conv3', reuse=True) as scope:
            w3 = tf.get_variable('weights')
        with tf.variable_scope('deconv4', reuse=True) as scope:
            w4 = tf.get_variable('weights')
        with tf.variable_scope('deconv5', reuse=True) as scope:
            w5 = tf.get_variable('weights')
        with tf.variable_scope('deconv6', reuse=True) as scope:
            w6 = tf.get_variable('weights')
        reg_loss = self.config.l2 * (tf.nn.l2_loss(w1)
                                     + tf.nn.l2_loss(w2)
                                     + tf.nn.l2_loss(w3)
                                     + tf.nn.l2_loss(w4)
                                     + tf.nn.l2_loss(w5)
                                     + tf.nn.l2_loss(w6))

        self.loss = ce_loss + reg_loss
