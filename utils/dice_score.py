import numpy as np
import tensorflow as tf


def dice_score(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    num = np.sum(np.logical_and(x_bool, y_bool)) * 2.
    denom = np.sum(x_bool) + np.sum(y_bool)
    return num / denom


def dice_score_tf(x, y):
    logical_and = tf.logical_and(x, y)
    logical_and = tf.cast(logical_and, tf.int32)
    num = 2 * tf.reduce_sum(logical_and)
    denom = tf.reduce_sum(tf.cast(x, tf.int32)) + tf.reduce_sum(tf.cast(y, tf.int32))
    return num / denom