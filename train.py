import numpy as np
import tensorflow as tf

from utils.general import Progbar
from utils.lr_schedule import LRSchedule


def train(model, debug):

    config = model.config

    ckpt_path = config.ckpt_path
    res_path = config.res_path

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                             start_decay=config.start_decay * len(train_ex_paths),
                             end_decay=config.end_decay * len(train_ex_paths),
                             lr_warm=config.lr_warm,
                             end_warm=config.end_warm * len(train_ex_paths))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
        best_fdice = 0

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
            _, _, _, fdice = model._segment(ex_path, sess)
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

            prog = Progbar(target=len(train_ex_paths))
            ex_bdices = []
            for ex, ex_path in enumerate(train_ex_paths):
                # print(ex_path)
                losses, bdices = model._train(ex_path, sess, lr_schedule.lr)
                train_losses.extend(losses)
                ex_bdices.append(np.mean(bdices))
                lr_schedule.update(batch_no=epoch * len(train_ex_paths) + ex)
                prog.update(ex + 1, values=[('loss', np.mean(losses))], exact=[("lr", lr_schedule.lr)])
                # print('******* Epoch %d Example %d: Training loss %5f' %(epoch, ex, np.mean(losses)))
            train_bdices.append(np.mean(ex_bdices))
            print('******************** Epoch %d: Training dice score %5f' %(epoch, np.mean(ex_bdices)))

            if epoch % 3 == 0:
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
                    _, _, _, fdice = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)
                    # print('******* Epoch %d Example %d: Test accuracy %5f' %(epoch, ex, fdice))
                val_fdices.append(np.mean(ex_fdices))
                print('******************** Epoch %d: Test dice score %5f' %(epoch, np.mean(ex_fdices)))

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
