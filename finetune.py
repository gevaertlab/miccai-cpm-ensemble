import numpy as np
import tensorflow as tf

from utils.general import Progbar
from utils.lr_schedule import LRSchedule


def finetune_all_layers(sess, model, train_ex_paths, lr):
    ex_bdices = []
    ex_losses = []
    prog = Progbar(target=len(train_ex_paths))
    for ex, ex_path in enumerate(train_ex_paths):
        losses, bdices = model._train(ex_path, sess, lr)
        ex_losses.extend(losses)
        ex_bdices.append(np.mean(bdices))
        prog.update(ex + 1, values=[('loss', np.mean(losses))], exact=[("lr", lr)])
    return ex_bdices, ex_losses


def finetune_last_layers(sess, model, train_ex_paths, lr):
    ex_bdices = []
    ex_losses = []
    prog = Progbar(target=len(train_ex_paths))
    for ex, ex_path in enumerate(train_ex_paths):
        losses, bdices = model._train_last_layers(ex_path, sess, lr)
        ex_losses.extend(losses)
        ex_bdices.append(np.mean(bdices))
        prog.update(ex + 1, values=[('loss', np.mean(losses))], exact=[("lr", lr)])
    return ex_bdices, ex_losses


def finetune_no_layers(sess, model, train_ex_paths):
    ex_bdices = []
    ex_losses = []
    for _, ex_path in enumerate(train_ex_paths):
        bdices = model._validate(ex_path, sess)
        ex_bdices.append(np.mean(bdices))
    return ex_bdices, ex_losses


def finetune(model, debug, detailed=False):
    config = model.config
    finetuning_method = config.finetuning_method

    ckpt_path = config.ckpt_path
    res_path = config.res_path

    train_ex_paths = model.train_ex_paths
    val_ex_paths = model.val_ex_paths
    if debug:
        train_ex_paths = train_ex_paths[:2]
        val_ex_paths = val_ex_paths[:2]

    saver = tf.train.Saver()

    lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                             start_decay=config.start_decay * len(train_ex_paths),
                             end_decay=config.end_decay * len(train_ex_paths),
                             lr_warm=config.lr_warm,
                             end_warm=config.end_warm * len(train_ex_paths))

    with tf.Session() as sess:

        saver.restore(sess, ckpt_path)

        train_losses = []
        train_bdices = []
        val_bdices = []
        val_fdices = []
        best_fdice = 0

        print('Initialization......')
        print('validate')
        ex_bdices = []
        for _, ex_path in enumerate(val_ex_paths):
            bdices = model._validate(ex_path, sess)
            ex_bdices.append(np.mean(bdices))
        val_bdices.append(np.mean(ex_bdices))
        print('******************** Initialization: Validation dice score %5f' %np.mean(ex_bdices))

        print('test')
        ex_fdices = []
        for _, ex_path in enumerate(val_ex_paths):
            _, _, _, fdice = model._segment(ex_path, sess)
            ex_fdices.append(fdice)
        val_fdices.append(np.mean(ex_fdices))
        print('******************** Initialization: Test dice score %5f' %np.mean(ex_fdices))

        if np.mean(ex_fdices) >= best_fdice:
            best_fdice = np.mean(ex_fdices)
            saver.save(sess, config.fine_tune_ckpt_path)

        for epoch in range(1, config.num_epochs + 1):
            print('epoch {}'.format(epoch))
            if finetuning_method == "all_layers":
                ex_bdices, ex_losses = finetune_all_layers(sess, model, train_ex_paths, lr_schedule.lr)
            elif finetuning_method == "last_layers":
                ex_bdices, ex_losses = finetune_last_layers(sess, model, train_ex_paths, lr_schedule.lr)
            elif finetuning_method == "no_layers":
                ex_bdices, ex_losses = finetune_no_layers(sess, model, train_ex_paths)
            else:
                print("Finetuning method not supported")
                raise NotImplementedError

            lr_schedule.update(batch_no=epoch * len(train_ex_paths))
            train_bdices.append(np.mean(ex_bdices))
            train_losses += ex_losses

            if epoch % 3 == 0:
                print('validate')
                ex_bdices = []
                for _, ex_path in enumerate(val_ex_paths):
                    bdices = model._validate(ex_path, sess)
                    ex_bdices.append(np.mean(bdices))
                val_bdices.append(np.mean(ex_bdices))
                print('******************** Epoch %d: Validation dice score %5f' %(epoch, np.mean(ex_bdices)))

                print('test')
                ex_fdices = []
                for _, ex_path in enumerate(val_ex_paths):
                    fy, fpred, fprob, fdice = model._segment(ex_path, sess)
                    ex_fdices.append(fdice)

                    if detailed and epoch == config.num_epochs:
                        fy = np.array(fy)
                        fpred = np.array(fpred)
                        fprob = np.array(fprob)
                        fdice = np.array(fdice)

                        ##########  example id for rembrandt
                        ex_id = ex_path[-7:-1]
                        ##########  example id for brats
                        # ex_id = ex_path.split('_')[-2] + '_' + ex_path.split('_')[-1]

                        ex_result_path = ex_path + '/pred.npz'
                        print('saving test result to %s' %(ex_result_path))
                        np.savez(ex_result_path,
                                 y=fy,
                                 pred=fpred,
                                 prob=fprob,
                                 dice=fdice)

                val_fdices.append(np.mean(ex_fdices))
                print('******************** Epoch %d: Test dice score %5f' %(epoch, np.mean(ex_fdices)))

                if np.mean(ex_fdices) >= best_fdice:
                    best_fdice = np.mean(ex_fdices)
                    saver.save(sess, config.fine_tune_ckpt_path)
                    print('Saving checkpoint to %s ......' %(config.fine_tune_ckpt_path))

        np.savez(res_path,
                 train_losses=train_losses,
                 train_bdices=train_bdices,
                 val_bdices=val_bdices,
                 val_fdices=val_fdices,
                 train_ex_paths=train_ex_paths,
                 val_ex_paths=val_ex_paths,
                 config=config.__dict__)
