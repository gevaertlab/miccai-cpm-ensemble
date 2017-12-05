import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data_utils import arr_to_im_path
from models.fcn import FCN_Model


def test(model, data_folder, results_folder):

    config = model.config
    ckpt_path = config.ckpt_path

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        for _, ex_path in enumerate(data_folder):
        	print('Running the prediction for patient {}'.format(ex_path))
            _, fpred, _, _ = model._segment(ex_path, sess)
            fpred = fpred.astype(int)
            out_path = results_folder + 'prediction.nii'
            arr_to_im_path(fpred, out_path)
            

if __name__ == '__main__':
    cfg_path = 'config_files/fcn_concat_finetune_rembrandt_v12.cfg'
    data_path = 'data/'
    results_path = 'data/results/'
    config = Config(cfg_path)
    model = FCN_Model(config)
    test(model, data_path, results_path)