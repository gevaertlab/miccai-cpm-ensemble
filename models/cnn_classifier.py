import os

import numpy as np
import tensorflow as tf

from models.model import Model
from utils.data_utils import get_ex_paths
from utils.dataset_v3 import get_dataset_single_patient_v3, get_dataset_batched
from utils.general import Progbar
from utils.lr_schedule import LRSchedule
from utils.metrics import all_scores


class CNN_Classifier(Model):
    def __init__(self, config):
        self.config = config
        self.patch = config.patch_size
        self.nb_classes = config.nb_classes
        self.nb_modalities = config.use_t1post + config.use_flair + config.use_segmentation

        self.load_data()
        self.add_dataset()
        self.add_placeholders()
        self.add_model()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()

    def load_data(self):
        # TODO revert :2
        self.train_ex_paths = get_ex_paths(self.config.train_path)
        self.val_ex_paths = get_ex_paths(self.config.val_path)

    def add_dataset(self):
        train_dataset = get_dataset_batched(self.config.train_path, False, self.config)

        train_nodrop_dataset = get_dataset_batched(self.config.train_path, True, self.config)
        val_dataset = get_dataset_batched(self.config.val_path, True, self.config)
        test_dataset = get_dataset_batched(self.config.test_path, True, self.config)

        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.data.Iterator.from_structure(
            output_types=(tf.float32,
                          tf.float32),
            output_shapes=([None, 320, 320, 24, self.nb_modalities],
                           [None, 1]))
        self.image, self.mgmtmethylated = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.train_nodrop_init_op = self.iterator.make_initializer(train_nodrop_dataset)
        self.val_init_op = self.iterator.make_initializer(val_dataset)
        self.test_init_op = self.iterator.make_initializer(test_dataset)


    def add_placeholders(self):
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[])
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])
        self.is_training = tf.placeholder(tf.bool, shape=[])

        # for tensorboard
        tf.summary.scalar("lr", self.lr_placeholder)

    def add_summary(self, sess):
        # tensorboard stuff
        # hardcoded
        # TODO: do it properly
        name_exp = self.config.res_path.strip().split('/')[1][:-4]
        summary_path = os.path.join('summaries', name_exp)
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(summary_path, sess.graph)

    def get_variables_to_restore(self):
        # to initialize some variables with pretained weights
        # 'level' refers to a level in the V-net architecture
        var_names_to_restore = ['conv1/conv3d/kernel:0',
                                'conv1/conv3d/bias:0',
                                'conv2/conv3d/kernel:0',
                                'conv2/conv3d/bias:0',
                                'conv3/conv3d/kernel:0',
                                'conv3/conv3d/bias:0',
                                'predict/dense/kernel:0',
                                'predict/dense/bias:0']

        var_to_restore = tf.contrib.framework.get_variables_to_restore(include=var_names_to_restore)
        var_to_train = tf.contrib.framework.get_variables_to_restore(exclude=var_names_to_restore)
        # print('*' * 50 + 'variables to retrain' + '*' * 50)
        # print([var.name for var in var_to_train])
        # print('*' * 50 + 'variables to restore' + '*' * 50)
        # print([var.name for var in var_to_restore])
        return var_to_train, var_to_restore

    def run_epoch(self, sess, lr_schedule, finetune=False):
        losses = []
        bdices = []
        batch = 0

        nbatches = len(self.train_ex_paths)
        prog = Progbar(target=nbatches)

        sess.run(self.train_init_op)

        while True:
            try:
                feed = {self.dropout_placeholder: self.config.dropout,
                        self.lr_placeholder: lr_schedule.lr,
                        self.is_training: self.config.use_batch_norm}

                if finetune:
                    pred, loss, summary, global_step, _ = sess.run([self.pred, self.loss,
                                                                    self.merged, self.global_step,
                                                                    self.train_last_layers],
                                                                   feed_dict=feed)
                else:
                    pred, loss, summary, global_step, _ = sess.run([self.pred, self.loss,
                                                                    self.merged, self.global_step,
                                                                    self.train],
                                                                   feed_dict=feed)
                batch += self.config.batch_size
            except tf.errors.OutOfRangeError:
                break

            if batch >= nbatches:
                break

            losses.append(loss)

            # logging
            prog.update(batch, values=[("loss", loss)], exact=[("lr", lr_schedule.lr),
                                                               ('score', lr_schedule.score)])
            # for tensorboard
            self.file_writer.add_summary(summary, global_step)

        return losses, np.mean(bdices)

    def run_test(self, sess):
        #sess.run(self.val_init_op)
        sess.run(self.test_init_op)
        #sess.run(self.train_nodrop_init_op)
        #sess.run(self.train_init_op)


        ypreds = []
        ytrues = []
        scores = []
        batch = 0

        nbatches = len(self.val_ex_paths)
        prog = Progbar(target=nbatches)
        print('\nValidation ...')
        while True:
            try:
                feed = {self.dropout_placeholder: 1.0,
                        self.is_training: False}

                pred, methylated, loss, score = sess.run([self.pred, self.mgmtmethylated, self.loss, self.score],
                                                  feed_dict=feed)

                scores.append(score)
                ypreds.extend(np.ravel(pred))
                ytrues.extend(np.ravel(methylated))

                batch += self.config.batch_size
                prog.update(batch)

            except tf.errors.OutOfRangeError:
                break

        
        return all_scores(ypred=ypreds, ytrue=ytrues), ytrues, scores

    def run_pred_single_example_v3(self, sess, patient):
        raise NotImplementedError

        if b'brats' in patient:
            name_dataset = 'Brats'
        elif b'TCGA' in patient:
            name_dataset = 'TCGA'
        else:
            name_dataset = 'not Brats'

        dataset = get_dataset_single_patient_v3(patient, self.config, name_dataset)
        init_op = self.iterator.make_initializer(dataset)
        sess.run(init_op)

        center = self.config.center_patch
        half_center = center // 2
        lower = self.patch // 2 - half_center
        fpred = None

        while True:
            try:
                feed = {self.dropout_placeholder: 1.0,
                        self.is_training: False}
                pat_shape, i, j, k, pred = sess.run([self.pat_shape, self.i, self.j, self.k, self.pred], feed_dict=feed)
            except tf.errors.OutOfRangeError:
                break

            if fpred is None:
                fpred = np.zeros(eval(pat_shape[0]))

            for idx, _ in enumerate(i):
                fpred[i[idx] - half_center:i[idx] + half_center,
                j[idx] - half_center:j[idx] + half_center,
                k[idx] - half_center:k[idx] + half_center] = pred[idx, lower:lower + center, \
                                                             lower:lower + center, lower:lower + center]

        return fpred

    def full_train(self, sess):
        config = self.config

        nbatches = len(self.train_ex_paths) * config.num_train_batches
        exp_decay = np.power(config.lr_min / config.lr_init,
                             1 / float(config.end_decay - config.start_decay))
        lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                                 start_decay=config.start_decay * nbatches,
                                 end_decay=config.end_decay * nbatches,
                                 lr_warm=config.lr_warm, decay_rate=config.decay_rate,
                                 end_warm=config.end_warm * nbatches, exp_decay=exp_decay)

        saver = tf.train.Saver()

        # for tensorboard
        self.add_summary(sess)

        precisions = []
        recalls = []
        f1s = []

        train_losses = []
        best_f1 = 0

        print('Start training ....')
        for epoch in range(1, config.num_epochs + 1):
            print('\nEpoch %d ...' % epoch)
            losses, train_dice = self.run_epoch(sess, lr_schedule)
            train_losses.extend(losses)

            if epoch % 2 == 0:
                precision, recall, f1 = self.run_test(sess)
                print('End of test, precision is %f, recall is %f and f1-score is %f' \
                      % (precision, recall, f1))
                # logging
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                lr_schedule.update(batch_no=epoch * nbatches, score=f1)

                if f1 >= best_f1:
                    best_f1 = f1

                    print('Saving checkpoint to %s ......' % (config.ckpt_path))
                    saver.save(sess, config.ckpt_path)

                    print('Saving results to %s ......' % (config.res_path))
                    np.savez(config.res_path,
                             train_losses=train_losses,
                             precisions=precisions,
                             recalls=recalls,
                             f1s=f1s,
                             train_ex_paths=self.train_ex_paths,
                             val_ex_paths=self.val_ex_paths,
                             config_file=config.__dict__)

            else:
                lr_schedule.update(batch_no=epoch * nbatches)

        return f1

    def add_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder) \
                .minimize(self.loss, global_step=self.global_step)

            if self.config.finetuning_method == 'last_layers':
                var_to_train, _ = self.get_variables_to_restore()
                self.train_last_layers = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder) \
                    .minimize(self.loss, var_list=var_to_train,
                              global_step=self.global_step)

    def add_model(self):
        # self.image = tf.reshape(self.image, [-1, self.patch, self.patch, self.patch, self.nb_modalities])
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
            relu1_1 = tf.nn.relu(conv1_1)

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
            relu1_2 = tf.nn.relu(conv1_2)

            # shape = (patch/2, patch/2, patch/2)
            pool1_2 = tf.layers.max_pooling3d(inputs=relu1_2, pool_size=(2, 2, 2),
                                              strides=(2, 2, 2), padding='VALID')

            drop1 = tf.nn.dropout(pool1_2, self.dropout_placeholder)

        with tf.variable_scope('conv2'):
            conv2_1 = tf.layers.conv3d(inputs=drop1,
                                       filters=2 * nb_filters,
                                       kernel_size=k_size,
                                       strides=(1, 1, 1),
                                       padding='SAME',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       kernel_regularizer=tf.nn.l2_loss)

            relu2_1 = tf.nn.relu(conv2_1)

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

            relu2_2 = tf.nn.relu(conv2_2)

            # shape = (patch/4, patch/4, patch/4)
            pool2_2 = tf.layers.max_pooling3d(inputs=relu2_2, pool_size=(2, 2, 2),
                                              strides=(2, 2, 2), padding='VALID')
            drop2 = tf.nn.dropout(pool2_2, self.dropout_placeholder)

        with tf.variable_scope('conv3'):
            conv3_1 = tf.layers.conv3d(inputs=drop2,
                                       filters=4 * nb_filters,
                                       kernel_size=k_size,
                                       strides=(1, 1, 1),
                                       padding='SAME',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       kernel_regularizer=tf.nn.l2_loss)

            relu3_1 = tf.nn.relu(conv3_1)
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

            relu3_2 = tf.nn.relu(conv3_2)

            # shape = (patch/8, patch/8, patch/8)
            pool3_2 = tf.layers.max_pooling3d(inputs=relu3_2, pool_size=(2, 2, 2),
                                              strides=(2, 2, 2), padding='VALID')
            drop3_2 = tf.nn.dropout(pool3_2, self.dropout_placeholder)

        with tf.variable_scope('conv4'):
            conv4_1 = tf.layers.conv3d(inputs=drop3_2,
                                       filters=8 * nb_filters,
                                       kernel_size=k_size,
                                       strides=(1, 1, 1),
                                       padding='SAME',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       kernel_regularizer=tf.nn.l2_loss)

            relu4_1 = tf.nn.relu(conv4_1)
            drop4_1 = tf.nn.dropout(relu4_1, self.dropout_placeholder)

            conv4_2 = tf.layers.conv3d(inputs=drop4_1,
                                       filters=8 * nb_filters,
                                       kernel_size=k_size,
                                       strides=(1, 1, 1),
                                       padding='SAME',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.0),
                                       kernel_regularizer=tf.nn.l2_loss)

            relu4_2 = tf.nn.relu(conv4_2)

            # shape = (patch/16, patch/16, patch/16)
            pool4_2 = tf.layers.max_pooling3d(inputs=relu4_2, pool_size=(2, 2, 2),
                                              strides=(2, 2, 2), padding='VALID')
            drop4_2 = tf.nn.dropout(pool4_2, self.dropout_placeholder)

            self.aggregate_features = tf.reduce_mean(drop4_2, axis=(1, 2, 3))

        with tf.variable_scope('predict'):
            self.score = tf.layers.dense(inputs=self.aggregate_features,
                                         units=1,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())

    def add_pred_op(self):
        self.pred = self.score >= 0

    def add_loss_op(self):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.mgmtmethylated)
        ce_loss = tf.reduce_mean(ce_loss)
        reg_loss = self.config.l2 * tf.losses.get_regularization_loss()

        self.loss = ce_loss + reg_loss

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
