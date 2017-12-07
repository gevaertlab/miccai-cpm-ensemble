import numpy as np
import tensorflow as tf

def test(model, debug, detailed=False):

    config = model.config

    ckpt_path = config.ckpt_path
    res_path = config.res_path

    val_ex_paths = model.val_ex_paths
    if debug:
        val_ex_paths = val_ex_paths[:2]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        print('Test')
        fdices_whole = []
        fdices_core = []
        fdices_enhancing = []
        for _, ex_path in enumerate(val_ex_paths):
            fy, fpred, fprob, dice_whole, dice_core, dice_enhancing = model._segment(ex_path, sess)
            fdices_whole.append(dice_whole)
            fdices_core.append(dice_core)
            fdices_enhancing.append(dice_enhancing)

            if detailed:
                fy = np.array(fy)
                fpred = np.array(fpred)
                fprob = np.array(fprob)
                fdice = np.array(fdice)

                ##########  example id for rembrandt
                # ex_id = ex_path[-7:-1]
                ##########  example id for brats
                ex_id = ex_path.split('_')[-2] + '_' + ex_path.split('_')[-1]

                ex_result_path = res_path + ex_id + '.npz'
                print('saving test result to %s' %(ex_result_path))
                np.savez(ex_result_path,
                         y=fy,
                         pred=fpred,
                         prob=fprob,
                         dice=fdice)

        np.savez(res_path,
                 fdices_whole=np.array(fdices_whole),
                 fdices_core=np.array(fdices_core),
                 fdices_enhancing=np.array(fdices_enhancing),
                 val_ex_paths=val_ex_paths,
                 config_file=config.__dict__)
