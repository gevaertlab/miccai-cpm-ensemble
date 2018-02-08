import os

import numpy as np
import tensorflow as tf

from models.fcn import FCN_Model
from utils.dataset import get_dataset
from utils.dataset import get_dataset_single_patient
from utils.dice_score import dice_score
from utils.dice_score import dice_score_tf
from utils.dice_score import get_inter_and_union
from utils.dice_score import dice_score_from_inters_and_unions
from utils.lr_schedule import LRSchedule
from utils.general import Progbar
from utils.data_utils import get_number_patches


class FCN_Concat(FCN_Model):

    def add_dataset(self):
        train_dataset = get_dataset(self.config.train_path, False, self.config)
        test_dataset = get_dataset(self.config.val_path, True, self.config)
        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.contrib.data.Iterator.from_structure(\
            output_types=(tf.string, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
            output_shapes=([None], [None], [None], [None],\
                           [None, self.patch, self.patch, self.patch, 4],\
                           [None, self.patch, self.patch, self.patch]))
        self.pat_path, self.i, self.j, self.k, self.image, self.label = self.iterator.get_next()

        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.test_init_op = self.iterator.make_initializer(test_dataset)

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

    def add_loss_op(self):
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.label)
        ce_loss = tf.reduce_mean(ce_loss)
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

        # dice score loss
        dice_loss = 0
        if self.config.use_dice_whole_loss:
            # dice score of WT
            whole_pred = tf.not_equal(self.pred, 0)
            whole_label = tf.not_equal(self.label, 0)
            ds_loss_whole = dice_score_tf(whole_pred, whole_label)
            dice_loss += tf.cast(self.config.ds_loss_beta * ds_loss_whole, tf.float32)

        if self.config.use_dice_core_loss:
            # dice score of TC
            core_pred = tf.logical_or(tf.equal(self.pred, 1), tf.equal(self.pred, 3))
            core_label = tf.logical_or(tf.equal(self.label, 1), tf.equal(self.label, 3))
            ds_loss_core = dice_score_tf(core_pred, core_label)
            # total dice score loss
            dice_loss += tf.cast(self.config.ds_loss_beta * ds_loss_core, tf.float32)

        if self.config.use_dice_enhancing_loss:
            # dice score of ET
            enhancing_pred = tf.equal(self.pred, 3)
            enhancing_label = tf.equal(self.label, 3)
            ds_loss_enhancing = dice_score_tf(enhancing_pred, enhancing_label)
            dice_loss += tf.cast(self.config.ds_loss_beta * ds_loss_enhancing, tf.float32)

        reg_loss = self.config.l2 * tf.losses.get_regularization_loss()

        self.loss = ce_loss + reg_loss + dice_loss

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
        batch = 0

        nbatches = len(self.train_ex_paths) * self.config.num_train_batches
        prog = Progbar(target=nbatches)

        sess.run(self.train_init_op)

        while True:
            try:
                feed = {self.dropout_placeholder: self.config.dropout,
                        self.lr_placeholder: lr_schedule.lr,
                        self.is_training: self.config.use_batch_norm}

                if finetune:
                    pred, loss, y, _ = sess.run([self.pred, self.loss, self.label, self.train_last_layers],\
                                                feed_dict=feed)
                else:
                    pred, loss, y, _ = sess.run([self.pred, self.loss, self.label, self.train],\
                                                feed_dict=feed)
                batch += 1
            except tf.errors.OutOfRangeError:
                break

            losses.append(loss)
            bdice = dice_score(y, pred)
            bdices.append(bdice)

            # logging
            prog.update(batch, values=[("loss", loss)], exact=[("lr", lr_schedule.lr),\
                                                               ('score', lr_schedule.score)])

        return losses, np.mean(bdices)

    def run_test(self, sess):
        sess.run(self.test_init_op)
        current_patient = ""

        all_dices_whole = []
        all_dices_core = []
        all_dices_enhancing = []

        HGG_patients = os.listdir('/labs/gevaertlab/data/tumor_segmentation/brats2017/HGG')
        HGG_patients = [os.path.join('/local-scratch/romain_scratch/brats2017/val', pat) for pat in HGG_patients]
        HGG_patients = [pat.encode('utf-8') for pat in HGG_patients]

        HGG_dices_whole = []
        HGG_dices_core = []
        HGG_dices_enhancing = []

        LGG_dices_whole = []
        LGG_dices_core = []
        LGG_dices_enhancing = []

        center = self.config.center_patch
        half_center = center // 2
        lower = self.patch // 2 - half_center

        nbatches = get_number_patches((155, 240, 240), self.patch, center) * len(self.val_ex_paths) // self.config.batch_size + 1
        prog = Progbar(target=nbatches)

        for batch in range(nbatches):
            feed = {self.dropout_placeholder: 1.0,
                    self.is_training: False}
            try:
                patients, i, j, k, y, pred = sess.run([self.pat_path, self.i, self.j, self.k, self.label, self.pred],
                                                      feed_dict=feed)
            except tf.errors.OutOfRangeError:
                break

            for idx, _ in enumerate(i):
                if patients[idx] != current_patient:
                    if current_patient != "":
                        # compute dice scores for different classes
                        # dice score for the Whole Tumor
                        dice_whole = dice_score(fy, fpred)
                        all_dices_whole.append(dice_whole)
                        if current_patient in HGG_patients:
                            HGG_dices_whole.append(dice_whole)
                        else:
                            LGG_dices_whole.append(dice_whole)
                        # print('dice score of whole of patient %s is %f'%(current_patient, dice_whole))

                        if self.nb_classes > 2:
                            # dice score for Tumor Core
                            fpred_core = (fpred == 1) + (fpred == 3)
                            fy_core = (fy == 1) + (fy == 3)
                            dice_core = dice_score(fy_core, fpred_core)
                            all_dices_core.append(dice_core)
                            if current_patient in HGG_patients:
                                HGG_dices_core.append(dice_core)
                            else:
                                LGG_dices_core.append(dice_core)
                            # print('dice score of core of patient %s is %f'%(current_patient, dice_core))

                            # dice score for Enhancing Tumor
                            if np.sum(fpred == 3) > 0:
                                fpred_enhancing = fpred == 3
                                fy_enhancing = fy == 3
                                dice_enhancing = dice_score(fy_enhancing, fpred_enhancing)
                                all_dices_enhancing.append(dice_enhancing)
                                if current_patient in HGG_patients:
                                    HGG_dices_enhancing.append(dice_enhancing)
                                else:
                                    LGG_dices_enhancing.append(dice_enhancing)
                            # print('dice score of enhancing of patient %s is %f'%(current_patient, dice_enhancing))

                    #hardcoded for BraTS
                    fpred = np.zeros((155, 240, 240))
                    fy = np.zeros((155, 240, 240))
                    current_patient = patients[idx]

                fy[i[idx] - half_center :i[idx] + half_center,
                   j[idx] - half_center:j[idx] + half_center,
                   k[idx] - half_center:k[idx] + half_center] = y[idx, :, :, :]
                fpred[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center] = pred[idx, lower:lower + center,\
                                                                        lower:lower + center, lower:lower + center]

            prog.update(batch + 1)

        return np.mean(all_dices_whole), np.mean(all_dices_core), np.mean(all_dices_enhancing),\
               np.mean(HGG_dices_whole), np.mean(HGG_dices_core), np.mean(HGG_dices_enhancing),\
               np.mean(LGG_dices_whole), np.mean(LGG_dices_core), np.mean(LGG_dices_enhancing)

    def run_test_v2(self, sess):
        sess.run(self.test_init_op)
        current_patient = ""

        all_dices_whole = []
        all_dices_core = []
        all_dices_enhancing = []

        # for Brats2017 only, because we don't have the split for Rembrandt
        HGG_patients = os.listdir('/labs/gevaertlab/data/tumor_segmentation/brats2017/HGG')
        HGG_patients = [os.path.join('/local-scratch/romain_scratch/brats2017/val', pat) for pat in HGG_patients]
        HGG_patients = [pat.encode('utf-8') for pat in HGG_patients]

        HGG_dices_whole = []
        HGG_dices_core = []
        HGG_dices_enhancing = []

        LGG_dices_whole = []
        LGG_dices_core = []
        LGG_dices_enhancing = []

        center = self.config.center_patch
        half_center = center // 2
        lower = self.patch // 2 - half_center

        while True:
            try:
                feed = {self.dropout_placeholder: 1.0, self.is_training: False}
                patients, y, pred = sess.run([self.pat_path, self.label, self.pred], feed_dict=feed)
            except tf.errors.OutOfRangeError:
                break

            for idx, pat in enumerate(patients):
                if pat != current_patient:
                    if current_patient != "":
                        # compute dice scores for different classes
                        # dice score for the Whole Tumor
                        dice_whole = dice_score_from_inters_and_unions(inters_whole, unions_whole)
                        all_dices_whole.append(dice_whole)
                        if current_patient in HGG_patients:
                            HGG_dices_whole.append(dice_whole)
                        else:
                            LGG_dices_whole.append(dice_whole)
                        # print('dice score of whole of patient %s is %f'%(current_patient, dice_whole))

                        if self.nb_classes > 2:
                            # dice score for Tumor Core
                            dice_core = dice_score_from_inters_and_unions(inters_core, unions_core)
                            all_dices_core.append(dice_core)
                            if current_patient in HGG_patients:
                                HGG_dices_core.append(dice_core)
                            else:
                                LGG_dices_core.append(dice_core)
                            # print('dice score of core of patient %s is %f'%(current_patient, dice_core))

                            # dice score for Enhancing Tumor
                            dice_enhancing = dice_score_from_inters_and_unions(inters_enhancing, unions_enhancing)
                            all_dices_enhancing.append(dice_enhancing)
                            if current_patient in HGG_patients:
                                HGG_dices_enhancing.append(dice_enhancing)
                            else:
                                LGG_dices_enhancing.append(dice_enhancing)
                            # print('dice score of enhancing of patient %s is %f'%(current_patient, dice_enhancing))

                    #hardcoded for BraTS
                    inters_enhancing = []
                    unions_enhancing = []
                    inters_core = []
                    unions_core = []
                    inters_whole = []
                    unions_whole = []
                    current_patient = pat

                crop_y = y[idx, :, :, :]
                crop_pred = pred[idx, lower:lower + center, lower:lower + center, lower:lower + center]

                # dice score for whole tumor
                in_wh, un_wh = get_inter_and_union(crop_pred, crop_y)
                inters_whole.append(in_wh)
                unions_whole.append(un_wh)

                if self.nb_classes > 2:
                    # dice score for tumor core
                    crop_pred_core = (crop_pred == 1) + (crop_pred == 3)
                    crop_y_core = (crop_y == 1) + (crop_y == 3)
                    in_co, un_co = get_inter_and_union(crop_pred_core, crop_y_core)
                    inters_core.append(in_co)
                    unions_core.append(un_co)

                    # dice score for tumor enhancing
                    crop_pred_enhancing = crop_pred == 3
                    crop_y_enhancing = crop_y == 3
                    in_en, un_en = get_inter_and_union(crop_pred_enhancing, crop_y_enhancing)
                    inters_enhancing.append(in_en)
                    unions_enhancing.append(un_en)

        return np.mean(all_dices_whole), np.mean(all_dices_core), np.mean(all_dices_enhancing),\
               np.mean(HGG_dices_whole), np.mean(HGG_dices_core), np.mean(HGG_dices_enhancing),\
               np.mean(LGG_dices_whole), np.mean(LGG_dices_core), np.mean(LGG_dices_enhancing)

    def run_test_single_example(self, sess, patient):
        dataset = get_dataset_single_patient(patient, self.config.batch_size, self.patch, self.config.center_patch)
        init_op = self.iterator.make_initializer(dataset)
        sess.run(init_op)

        #hardcoded for BraTS
        center = self.config.center_patch
        half_center = center // 2
        lower = self.patch // 2 - half_center
        fpred = np.zeros((155, 240, 240))

        while True:
            try:
                feed = {self.dropout_placeholder: 1.0,
                        self.is_training: False}
                i, j, k, pred = sess.run([self.i, self.j, self.k, self.pred], feed_dict=feed)
            except tf.errors.OutOfRangeError:
                break

            for idx, _ in enumerate(i):
                fpred[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center] = pred[idx, lower:lower + center,\
                                                                        lower:lower + center, lower:lower + center]

        return fpred

    def full_train(self, sess):
        config = self.config

        nbatches = len(self.train_ex_paths) * config.num_train_batches
        exp_decay = np.power(config.lr_min / config.lr_init,\
                             1 / float(config.end_decay - config.start_decay))
        lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                                 start_decay=config.start_decay * nbatches,
                                 end_decay=config.end_decay * nbatches,
                                 lr_warm=config.lr_warm, decay_rate=config.decay_rate,
                                 end_warm=config.end_warm * nbatches, exp_decay=exp_decay)

        saver = tf.train.Saver()

        train_losses = []
        train_bdices = []
        test_whole_dices = []
        test_core_dices = []
        test_enhancing_dices = []
        best_fdice = 0

        print('Start training ....')
        for epoch in range(1, config.num_epochs + 1):
            print('Epoch %d ...'%epoch)
            losses, train_dice = self.run_epoch(sess, lr_schedule)
            train_losses.extend(losses)
            train_bdices.append(train_dice)

            if epoch % 2 == 0:
                test_whole, test_core, test_enhancing, _, _, _, _, _, _ = self.run_test(sess)
                print('End of test, whole dice score is %f, core dice score is %f and enhancing dice score is %f'\
                      %(test_whole, test_core, test_enhancing))
                # logging
                test_whole_dices.append(test_whole)
                test_core_dices.append(test_core)
                test_enhancing_dices.append(test_enhancing)
                lr_schedule.update(batch_no=epoch * nbatches, score=test_core + test_enhancing)

                if test_core + test_enhancing >= best_fdice:
                    best_fdice = test_core + test_enhancing

                    print('Saving checkpoint to %s ......' %(config.ckpt_path))
                    saver.save(sess, config.ckpt_path)

                    print('Saving results to %s ......'%(config.res_path))
                    np.savez(config.res_path,
                             train_losses=train_losses,
                             train_bdices=train_bdices,
                             test_whole_dices=test_whole_dices,
                             test_core_dices=test_core_dices,
                             test_enhancing_dices=test_enhancing_dices,
                             train_ex_paths=self.train_ex_paths,
                             val_ex_paths=self.val_ex_paths,
                             config_file=config.__dict__)

            else:
                lr_schedule.update(batch_no=epoch * nbatches)

        return test_whole

    def finetune(self, sess):
        saver = tf.train.Saver()

        print('Initializing / restoring weights ...')
        if self.config.finetuning_method == "all_layers":
            saver.restore(sess, self.config.ckpt_path_to_finetune)
        elif self.config.finetuning_method == "no_layers":
            sess.run(tf.global_variables_initializer())
        else:
            print("Finetuning method not supported")
            raise NotImplementedError

        return self.full_train(sess)
        