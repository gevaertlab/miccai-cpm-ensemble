import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data_utils import arr_to_im_path
from models.fcn_concat import FCN_Concat


def test(model, patient):

    config = model.config
    ckpt_path = config.ckpt_path

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        fpred = model.run_test_single_example(sess, patient)
        fpred[fpred == 3] = 4
        out_path = patient + 'results/tumor_gevaertmodel_class.nii'
        arr_to_im_path(fpred, out_path)
            

if __name__ == '__main__':
    cfg_path = 'config_files/fcn_train_concat_2017.cfg'
    data_path = 'data/'
    config = Config(cfg_path)
    config.val_path = data_path
    model = FCN_Concat(config)
    test(model, data_path)