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


def get_inter_and_union(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    inter = np.sum(np.logical_and(x_bool, y_bool)) * 2.
    union = np.sum(x_bool) + np.sum(y_bool)
    return inter, union


def dice_score_from_inters_and_unions(inters, unions):
    sum_inters = np.sum(inters)
    sum_unions = np.sum(unions)
    if sum_unions == 0:
        return 1
    return sum_inters / sum_unions
