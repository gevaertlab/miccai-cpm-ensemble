import threading

import numpy as np
import tensorflow as tf

from utils.general import Progbar
from utils.lr_schedule import LRSchedule
from utils.data_iterator import fcn_data_iter_v2

def load_and_enqueue(sess, model, coord):
    bs = model.config.batch_size
    nb = model.config.num_train_batches
    patch_size = model.patch
    all_paths = model.train_ex_paths
    while not coord.should_stop():
        for ex_path in all_paths:
            for x, y in fcn_data_iter_v2(ex_path, 'fgbg', bs, nb, patch_size):
                sess.run(model.enqueue_op, feed_dict={model.image_batch_input: x,
                                                      model.label_batch_input: y})

def train_v2(model, debug):

    config = model.config

    ckpt_path = config.ckpt_path
    res_path = config.res_path

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    batch_size = config.batch_size
    num_train_batches = config.num_train_batches
    nb_batch_per_epoch = len(train_ex_paths) * num_train_batches
    patch_size = config.patch_size

    lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                             start_decay=config.start_decay * nb_batch_per_epoch,
                             end_decay=config.end_decay * nb_batch_per_epoch,
                             lr_warm=config.lr_warm, decay_rate=config.decay_rate,
                             end_warm=config.end_warm * nb_batch_per_epoch)

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
        best_fdice = 0

        print('run the queue for preprocessing of training data ...')
        t = threading.Thread(target=load_and_enqueue, args=(sess, model, coord))
        t.start()

        print('Initialization......')
        print('validate')
        ex_bdices = []
        for ex, ex_path in enumerate(val_ex_paths):
            # print(ex_path)
            bdices = model._validate(ex_path, sess)
            ex_bdices.append(np.mean(bdices))
            # average dice score for 20 batches of every sample
            # print('******* Epoch %d Example %d: Validation accuracy %5f' %(0, ex, np.mean(bdices)))
        val_bdices.append(np.mean(ex_bdices))
        # average dice score for all samples
        print('******************** Epoch %d: Validation dice score %5f' %(0, np.mean(ex_bdices)))

        print('test')
        ex_fdices = []
        for ex, ex_path in enumerate(val_ex_paths):
            # print(ex_path)
            _, _, _, fdice, _, _ = model._segment(ex_path, sess)
            ex_fdices.append(fdice)
            # full dice score per sample
            # print('******* Epoch %d Example %d: Test accuracy %5f' %(0, ex, fdice))
        val_fdices.append(np.mean(ex_fdices))
        # average full dice score for all samples
        print('******************** Epoch %d: Test dice score %5f' %(0, np.mean(ex_fdices)))

        if np.mean(ex_fdices) >= best_fdice:
            best_fdice = np.mean(ex_fdices)
            saver.save(sess, ckpt_path)

        for epoch in range(1, config.num_epochs+1):
            print('\nepoch {}'.format(epoch))
            print('train')

            prog = Progbar(target=nb_batch_per_epoch)
            ex_bdices = []

            for batch_no in range(nb_batch_per_epoch):
                loss, bdice = model._train_v2(sess, lr_schedule.lr)
                train_losses.append(loss)
                ex_bdices.append(bdice)
                lr_schedule.update(batch_no=epoch * nb_batch_per_epoch + batch_no)
                prog.update(batch_no, values=[('loss', loss)], exact=[("lr", lr_schedule.lr)])
            train_bdices.append(np.mean(ex_bdices))
            print('******************** Epoch %d: Training dice score %5f' %(epoch, np.mean(ex_bdices)))

            if epoch % 2 == 0:
                print('validate')
                ex_bdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    bdices = model._validate(ex_path, sess)
                    ex_bdices.append(np.mean(bdices))
                    # print('******* Epoch %d Example %d: Validation accuracy %5f' %(epoch, ex, np.mean(bdices)))
                val_bdices.append(np.mean(ex_bdices))
                print('******************** Epoch %d: Validation dice score %5f' %(epoch, np.mean(ex_bdices)))

                print('test')
                ex_fdices = []
                for ex, ex_path in enumerate(val_ex_paths):
                    # print(ex_path)
                    _, _, _, fdice, _, _ = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)
                    # print('******* Epoch %d Example %d: Test accuracy %5f' %(epoch, ex, fdice))
                val_fdices.append(np.mean(ex_fdices))
                print('******************** Epoch %d: Test dice score %5f' %(epoch, np.mean(ex_fdices)))
                lr_schedule.update(score=np.mean(ex_fdices))

                if np.mean(ex_fdices) >= best_fdice:
                    best_fdice = np.mean(ex_fdices)
                    saver.save(sess, ckpt_path)
                    print('Saving checkpoint to %s ......' %(ckpt_path))

        print('Saving losses to %s........' %(res_path))
        np.savez(res_path,
        			train_losses=train_losses,
             	  	train_bdices=train_bdices,
             	  	val_bdices=val_bdices,
             	  	val_fdices=val_fdices,
             	  	train_ex_paths=train_ex_paths,
             	  	val_ex_paths=val_ex_paths,
             	  	config_file=config.__dict__)

        # stop queue
        coord.request_stop()
        coord.join([t])
