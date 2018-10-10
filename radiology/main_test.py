import os

import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data_utils import arr_to_im_path
from models.fcn_concat import FCN_Concat


# this file is for the docker container of the model
def run_test(model, patient):
    config = model.config
    ckpt_path = '/gevaertlab/' + config.ckpt_path

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        fpred = model.run_pred_single_example_v3(sess, patient)
        fpred[fpred == 3] = 4
        out_path = patient.decode('utf-8') + 'results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path += 'tumor_gevaertlab_class.nii'
        arr_to_im_path(fpred, out_path)
            

if __name__ == '__main__':
    cfg_path = '/gevaertlab/config_files/fcn_train_concat_2017_v66.cfg'
    data_path = b'/data/'
    config = Config(cfg_path)
    model = FCN_Concat(config)
    run_test(model, data_path)
