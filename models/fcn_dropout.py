import tensorflow as tf

from models.fcn import FCN_Model

class FCNDropoutModel(FCN_Model):

    def add_model(self):
        batch_size = tf.shape(self.label_placeholder)[0]

        with tf.variable_scope('conv1') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 4, 10],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                    initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(input=self.image_placeholder, filter=kernel, strides = [1,1,1,1,1], padding="SAME")
            relu1 = tf.nn.relu(conv + bias)

            
            pool1 = tf.layers.max_pooling3d(inputs = relu1, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')
            drop1 = tf.nn.dropout(pool1, self.dropout_placeholder)

            # print(conv1.get_shape())
            # print(pool1.get_shape())

        with tf.variable_scope('conv2') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [20],
                    initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv3d(input=drop1, filter=kernel, strides = [1,1,1,1,1], padding="SAME")
            relu2 = tf.nn.relu(conv + bias)
            
            pool2 = tf.layers.max_pooling3d(inputs = relu2, pool_size = (2,2,2), strides = (2,2,2), padding='VALID')
            drop2 = tf.nn.dropout(pool2, self.dropout_placeholder)
            # print(conv2.get_shape())
            # print(pool2.get_shape())


        with tf.variable_scope('deconv3') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 10, 20],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [10],
                    initializer=tf.constant_initializer(0.0))

            deconv = tf.nn.conv3d_transpose(drop2, filter = kernel, output_shape = [batch_size, 12, 12, 12, 10], strides = [1,2,2,2,1], padding='SAME')
            
            relu3 = tf.nn.relu(deconv + bias)    
            drop3 = tf.nn.dropout(relu3, self.dropout_placeholder)

            # print(deconv3.get_shape())
            # print(relu3.get_shape())

        with tf.variable_scope('deconv4') as scope:

            kernel = tf.get_variable('weights', [5, 5, 5, 2, 10],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', [2],
                    initializer=tf.constant_initializer(0.0))

            deconv4 = tf.nn.conv3d_transpose(drop3, filter = kernel, output_shape = [batch_size, 24, 24, 24, 2], strides = [1,2,2,2,1], padding='SAME')

            # print(deconv4.get_shape())
            # print(relu4.get_shape())

            self.score = deconv4 + bias

