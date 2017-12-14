import numpy as np
import tensorflow as tf

from models.fcn import FCN_Model
from utils.dataset import get_dataset
from utils.dice_score import dice_score


class FCN_Concat(FCN_Model):

    def add_dataset(self):
        train_dataset = get_dataset(self.config.train_path, False, self.config.batch_size, self.patch)
        val_dataset = get_dataset(self.config.val_path, False, self.config.batch_size, self.patch)
        test_dataset = get_dataset(self.config.val_path, True, self.config.batch_size, self.patch)

        # iterator just needs to know the output types and shapes of the datasets
        iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                           train_dataset.output_shapes)
        self.pat_path, self.i, self.j, self.k, self.image, self.label = iterator.get_next()

        self.train_init_op = iterator.make_initializer(train_dataset)
        self.val_init_op = iterator.make_initializer(val_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

    def add_model(self):
        self.image = tf.reshape(self.image, [-1, self.patch, self.patch, self.patch, 4])
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(inputs=self.image,
                                     filters=10,
                                     kernel_size=5,
                                     strides=(1, 1, 1),
                                     padding='SAME',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0))
            # TODO: uncomment to use queues
            # conv1 = tf.layers.conv3d(inputs=self.image_batch,
            #                          filters=10,
            #                          kernel_size=5,
            #                          strides=(1, 1, 1),
            #                          padding='SAME',
            #                          activation=None,
            #                          use_bias=True,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                          bias_initializer=tf.constant_initializer(0.0))

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
                                                 filters=self.nb_classes,
                                                 kernel_size=5,
                                                 strides=(2, 2, 2),
                                                 padding='SAME',
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))

            # print(deconv6.get_shape())
            bias = tf.get_variable('biases', [self.nb_classes],
                                   initializer=tf.zeros_initializer()) 
            self.score = deconv6 + bias

    def add_loss_op(self):
        logits = tf.reshape(self.score, [-1, self.nb_classes])
        labels = tf.reshape(self.label, [-1])
        # labels = tf.reshape(self.label_batch, [-1])

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

    def run_epoch(self, sess, lr_schedule, finetune=False):
        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches
        nbatches = nb * bs

        prog = Progbar(target=nbatches)

        sess.run(self.train_init_op)

        for batch in range(nbatches):
            feed = {self.dropout_placeholder: self.config.dropout,
                    self.lr_placeholder: lr_schedule.lr,
                    self.is_training: self.config.use_batch_norm}
            if finetune:
                pred, loss, y, _ = sess.run([self.pred, self.loss, self.label, self.train_last_layers], feed_dict=feed)
            else:
                pred, loss, y, _ = sess.run([self.pred, self.loss, self.label, self.train], feed_dict=feed)

            losses.append(loss)
            bdice = dice_score(y, pred)
            bdices.append(bdice)

            # logging
            prog.update(batch + 1, values=[("loss", loss)], exact=[("lr", lr_schedule.lr)])

        return losses, np.mean(bdices)
    
    def run_evaluate(self, sess):
        bs = self.config.batch_size
        nb = self.config.num_val_batches
        nbatches = nb * bs

        bdices = []

        prog = Progbar(target=nbatches)

        sess.run(self.val_init_op)

        for batch in range(nbatches):
            feed = {self.dropout_placeholder: 1,
                    self.is_training: False}
            pred, y = sess.run([self.pred, self.label], feed_dict=feed)
            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return np.mean(bdices)

    def run_test(self, sess):
        # hardcoded for BraTS: they are 5292 patches per volume
        nbatches = int(len(self.val_ex_paths) * 5292 / 50) + 1
        sess.run(self.test_init_op)
        current_patient = ""

        all_dices_whole = []
        all_dices_core = []
        all_dices_enhancing = []

        for _ in range(nbatches):
            feed = {self.dropout_placeholder: 1.0,
                    self.is_training:False}
            patients, i, j, k, y, pred, prob = sess.run([self.pat_path, self.i, self.j,\
                                                        self.k, self.label, self.pred, self.prob],
                                                        feed_dict=feed)

            for idx, _ in enumerate(i):
                if patients[idx] != current_patient:
                    if current_patient != "":
                        # compute dice scores for different classes
                        # dice score for the Whole Tumor
                        dice_whole = dice_score(fy, fpred)
                        all_dices_whole.append(dice_whole)
                        print('dice score of whole of patient %s is %d'%(current_patient, dice_whole))

                        if self.nb_classes > 2:
                            # dice score for Tumor Core
                            fpred_core = (fpred == 1) + (fpred == 3)
                            fy_core = (fy == 1) + (fy == 3)
                            dice_core = dice_score(fy_core, fpred_core)
                            all_dices_core.append(dice_core)
                            print('dice score of core of patient %s is %d'%(current_patient, dice_core))

                            # dice score for Enhancing Tumor
                            fpred_enhancing = fpred == 3
                            fy_enhancing = fy == 3
                            dice_enhancing = dice_score(fy_enhancing, fpred_enhancing)
                            all_dices_enhancing.append(dice_enhancing)
                            print('dice score of enhancing of patient %s is %d'%(current_patient, dice_enhancing))

                    #hardcoded for BraTS
                    fpred = np.zeros((155, 240, 240))
                    fy = np.zeros((155, 240, 240))
                    fprob = np.zeros((155, 240, 240, 2))
                    current_patient = patients[idx]

                fy[i[idx]:i[idx] + self.patch,
                   j[idx]:j[idx] + self.patch,
                   k[idx]:k[idx] + self.patch] = y[idx, :, :, :]
                fpred[i[idx]:i[idx] + self.patch,
                      j[idx]:j[idx] + self.patch,
                      k[idx]:k[idx] + self.patch] = pred[idx, :, :, :]
                fprob[i[idx]:i[idx] + self.patch,
                      j[idx]:j[idx] + self.patch,
                      k[idx]:k[idx] + self.patch, :] = prob[idx, :, :, :, :]

        return all_dices_whole

    def train(self, sess, lr_schedule):
        nbatches = self.config.batch_size * self.config.num_train_batches

        for epoch in range(self.config.nb_epochs):
            _, train_dice = self.run_epoch(sess, lr_schedule)
            if epoch % 2 == 0:
                val_dice = self.run_evaluate(self, sess)
                # TODO: compute test_dice score and update score of lr_schedule with test_dice
                lr_schedule.update(batch_no=epoch * nbatches, score=val_dice)
            else:
                lr_schedule.update(batch_no=epoch * nbatches)
