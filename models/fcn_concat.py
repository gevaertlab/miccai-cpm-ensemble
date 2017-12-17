import numpy as np
import tensorflow as tf

from models.fcn import FCN_Model
from utils.dataset import get_dataset
from utils.dataset import get_dataset_single_patient
from utils.dice_score import dice_score
from utils.lr_schedule import LRSchedule
from utils.general import Progbar
from utils.data_utils import get_number_patches


class FCN_Concat(FCN_Model):

    def add_dataset(self):
        train_dataset = get_dataset(self.config.train_path, False, self.config.batch_size,\
                                    self.patch, self.config.center_patch)
        val_dataset = get_dataset(self.config.val_path, False, self.config.batch_size,\
                                  self.patch, self.config.center_patch)
        test_dataset = get_dataset(self.config.val_path, True, self.config.batch_size,\
                                   self.patch, self.config.center_patch)

        # iterator just needs to know the output types and shapes of the datasets
        self.iterator = tf.contrib.data.Iterator.from_structure(\
            output_types=(tf.string, tf.int32, tf.int32,tf.int32, tf.float32, tf.int32),
            output_shapes=([None], [None], [None], [None],\
                           [None, self.patch, self.patch, self.patch, 4],\
                           [None, self.patch, self.patch, self.patch]))
        self.pat_path, self.i, self.j, self.k, self.image, self.label = self.iterator.get_next()

        self.train_init_op = self.iterator.make_initializer(train_dataset)
        self.val_init_op = self.iterator.make_initializer(val_dataset)
        self.test_init_op = self.iterator.make_initializer(test_dataset)

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
                                     bias_initializer=tf.constant_initializer(0.0))
            # TODO: uncomment to use queues
            # conv1 = tf.layers.conv3d(inputs=self.image_batch,
            #                          filters=10,
            #                          kernel_size=k_size,
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
                                     filters=2 * nb_filters,
                                     kernel_size=k_size,
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
                                     filters=4 * nb_filters,
                                     kernel_size=k_size,
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
                                                 filters=2 * nb_filters,
                                                 kernel_size=k_size,
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
            print('Shape of bn4 is:', bn4.get_shape())
            print('shape of pool2 is:', pool2.get_shape())
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
                                                 bias_initializer=tf.constant_initializer(0.0))
            bias = tf.get_variable('biases', [10],
                                   initializer=tf.zeros_initializer())
            deconv5 = deconv5 + bias
            # shape = (patch/2, patch/2, patch/2)
            bn5 = tf.layers.batch_normalization(deconv5, axis=-1, training=self.is_training)
            print('Shape of bn5 is:', bn5.get_shape())
            print('shape of pool1 is:', pool1.get_shape())
            
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

        nb = self.config.num_train_batches
        # TODO: so far take only 10 batches per image for memory issue
        nbatches = 10 * len(self.train_ex_paths)

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
            prog.update(batch + 1, values=[("loss", loss)], exact=[("lr", lr_schedule.lr), ('score', lr_schedule.score)])

        return losses, np.mean(bdices)
    
    def run_evaluate(self, sess):
        nb = self.config.num_val_batches
        #TODO: same as above
        nbatches = 10 * len(self.val_ex_paths)

        bdices = []

        prog = Progbar(target=nbatches)

        sess.run(self.val_init_op)

        for batch in range(nbatches):
            feed = {self.dropout_placeholder: 1,
                    self.is_training: False}
            pred, y = sess.run([self.pred, self.label], feed_dict=feed)
            bdice = dice_score(y, pred)
            bdices.append(bdice)

            prog.update(batch + 1, values=[("dice score", bdice)])


        return np.mean(bdices)

    def run_test(self, sess):
        # hardcoded for BraTS: they are 196 patches per volume
        # nbatches = int(len(self.val_ex_paths) * 196 / 50) + 1
        sess.run(self.test_init_op)
        current_patient = ""

        all_dices_whole = []
        all_dices_core = []
        all_dices_enhancing = []

        # for _ in range(nbatches):
        while True:
            try:
                feed = {self.dropout_placeholder: 1.0,
                        self.is_training:False}
                patients, i, j, k, y, pred, prob = sess.run([self.pat_path, self.i, self.j,\
                                                            self.k, self.label, self.pred, self.prob],
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
                        # print('dice score of whole of patient %s is %f'%(current_patient, dice_whole))

                        if self.nb_classes > 2:
                            # dice score for Tumor Core
                            fpred_core = (fpred == 1) + (fpred == 3)
                            fy_core = (fy == 1) + (fy == 3)
                            dice_core = dice_score(fy_core, fpred_core)
                            all_dices_core.append(dice_core)
                            # print('dice score of core of patient %s is %f'%(current_patient, dice_core))

                            # dice score for Enhancing Tumor
                            fpred_enhancing = fpred == 3
                            fy_enhancing = fy == 3
                            dice_enhancing = dice_score(fy_enhancing, fpred_enhancing)
                            all_dices_enhancing.append(dice_enhancing)
                            # print('dice score of enhancing of patient %s is %f'%(current_patient, dice_enhancing))

                    #hardcoded for BraTS
                    fpred = np.zeros((155, 240, 240))
                    fy = np.zeros((155, 240, 240))
                    fprob = np.zeros((155, 240, 240, 4))
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

        return np.mean(all_dices_whole), np.mean(all_dices_core), np.mean(all_dices_enhancing)

    def run_test_v2(self, sess):
        sess.run(self.test_init_op)
        current_patient = ""

        all_dices_whole = []
        all_dices_core = []
        all_dices_enhancing = []

        center = self.config.center_patch
        half_center = center // 2
        lower = self.patch // 2 - half_center

        nbatches = get_number_patches((155, 240, 240), self.patch, center) * len(self.val_ex_paths) / self.config.batch_size + 1
        prog = Progbar(target=nbatches)

        for batch in range(nbatches):
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
                        # print('dice score of whole of patient %s is %f'%(current_patient, dice_whole))

                        if self.nb_classes > 2:
                            # dice score for Tumor Core
                            fpred_core = (fpred == 1) + (fpred == 3)
                            fy_core = (fy == 1) + (fy == 3)
                            dice_core = dice_score(fy_core, fpred_core)
                            all_dices_core.append(dice_core)
                            # print('dice score of core of patient %s is %f'%(current_patient, dice_core))

                            # dice score for Enhancing Tumor
                            fpred_enhancing = fpred == 3
                            fy_enhancing = fy == 3
                            dice_enhancing = dice_score(fy_enhancing, fpred_enhancing)
                            all_dices_enhancing.append(dice_enhancing)
                            # print('dice score of enhancing of patient %s is %f'%(current_patient, dice_enhancing))

                    #hardcoded for BraTS
                    fpred = np.zeros((155, 240, 240))
                    fy = np.zeros((155, 240, 240))
                    fprob = np.zeros((155, 240, 240, 4))
                    current_patient = patients[idx]

                fy[i[idx] - half_center :i[idx] + half_center,
                   j[idx] - half_center:j[idx] + half_center,
                   k[idx] - half_center:k[idx] + half_center] = y[idx, :, :, :]
                fpred[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center] = pred[idx, lower:lower + center,\
                                                                        lower:lower + center, lower:lower + center]
                fprob[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center, :] = prob[idx, lower:lower + center,\
                                                                           lower:lower + center, lower:lower + center, :]

            prog.update(batch + 1)

        return np.mean(all_dices_whole), np.mean(all_dices_core), np.mean(all_dices_enhancing)

    def run_test_single_example(self, sess, patient):
        dataset = get_dataset_single_patient(patient, self.config.batch_size, self.patch, self.config.center_patch)
        init_op = self.iterator.make_initializer(dataset)
        sess.run(init_op)


        #hardcoded for BraTS
        half_center = 5
        fpred = np.zeros((155, 240, 240))
        fy = np.zeros((155, 240, 240))
        fprob = np.zeros((155, 240, 240, 4))

        while True:
            try:
                feed = {self.dropout_placeholder: 1.0,
                        self.is_training:False}
                i, j, k, y, pred, prob = sess.run([self.i, self.j, self.k, self.label, self.pred, self.prob],
                                                  feed_dict=feed)
            except tf.errors.OutOfRangeError:
                break

            for idx, _ in enumerate(i):
                fy[i[idx] - half_center :i[idx] + half_center,
                   j[idx] - half_center:j[idx] + half_center,
                   k[idx] - half_center:k[idx] + half_center] = y[idx, :, :, :]
                fpred[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center] = pred[idx, 12:22, 12:22, 12:22]
                fprob[i[idx] - half_center:i[idx] + half_center,
                      j[idx] - half_center:j[idx] + half_center,
                      k[idx] - half_center:k[idx] + half_center, :] = prob[idx, 12:22, 12:22, 12:22, :]

        # dice score for the Whole Tumor
        dice_whole = dice_score(fy, fpred)
        print('dice score of whole of patient %s is %f'%(patient, dice_whole))

        # dice score for Tumor Core
        fpred_core = (fpred == 1) + (fpred == 3)
        fy_core = (fy == 1) + (fy == 3)
        dice_core = dice_score(fy_core, fpred_core)
        print('dice score of core of patient %s is %f'%(patient, dice_core))

        # dice score for Enhancing Tumor
        fpred_enhancing = fpred == 3
        fy_enhancing = fy == 3
        dice_enhancing = dice_score(fy_enhancing, fpred_enhancing)
        print('dice score of enhancing of patient %s is %f'%(patient, dice_enhancing))

        return fpred

    def full_train(self, sess):
        config = self.config

        nbatches = config.batch_size * config.num_train_batches
        ckpt_path = config.ckpt_path
        res_path = config.res_path

        lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                                 start_decay=config.start_decay * len(self.train_ex_paths),
                                 end_decay=config.end_decay * len(self.train_ex_paths),
                                 lr_warm=config.lr_warm,
                                 end_warm=config.end_warm * len(self.train_ex_paths))

        saver = tf.train.Saver()

        train_losses = []
        train_bdices = []
        val_bdices = []
        test_whole_dices = []
        test_core_dices = []
        test_enhancing_dices = []
        best_fdice = 0

        print('Start training ....')
        for epoch in range(1, config.num_epochs + 1):
            print('Epoch %d ...'%epoch)
            losses, train_dice = self.run_epoch(sess, lr_schedule)
            train_losses.extend(losses)
            train_bdices.append(train_bdices)

            if epoch % 3 == 0:
                val_dice = self.run_evaluate(sess)
                print('End of evaluation, validation dice score is:', val_dice)
                test_whole, test_core, test_enhancing = self.run_test_v2(sess)
                print('End of test, test dice score is:', test_whole)
                # logging
                val_bdices.append(val_dice)
                test_whole_dices.append(test_whole)
                test_core_dices.append(test_core)
                test_enhancing_dices.append(test_enhancing)
                lr_schedule.update(batch_no=epoch * nbatches, score=test_whole)

                if test_whole >= best_fdice:
                    best_fdice = test_whole
                    saver.save(sess, ckpt_path)
                    print('Saving checkpoint to %s ......' %(ckpt_path))

                np.savez(res_path,
                         train_losses=train_losses,
                         train_bdices=train_bdices,
                         val_bdices=val_bdices,
                         test_whole_dices=test_whole_dices,
                         test_core_dices=test_core_dices,
                         test_enhancing_dices=test_enhancing_dices,
                         train_ex_paths=self.train_ex_paths,
                         val_ex_paths=self.val_ex_paths,
                         config_file=config.__dict__)

            else:
                lr_schedule.update(batch_no=epoch * nbatches)

        return test_whole
