import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from keras.backend.tensorflow_backend import set_session
from config import Config
from models import Model
import matplotlib
matplotlib.use('Agg')


if os.path.isdir("output/"):
    shutil.rmtree("output/")
os.makedirs("output/")

config = Config(epochs = 30, gpu = "1", sampling_size_train = 5, sampling_size_val = 5, batch_size = 12 ,lr = 1e-4, val_size = 0.25)

session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = config.gpu
session_config.gpu_options.allow_growth = True
set_session(tf.Session(config=session_config))

model = Model(config)
y_scores, y_preds = model.train_predict()
model.get_metrics(y_scores, y_preds)
#model.plot_ROCs(y_scores)
#model.plot_PRs(y_scores)