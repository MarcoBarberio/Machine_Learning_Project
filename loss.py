import tensorflow as tf
import numpy as np

def mee(y_true, y_pred):
    diff = y_true - y_pred
    eucl = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1))
    return tf.reduce_mean(eucl)


def mee_np(y_true, y_pred):
    diff = y_true - y_pred
    eucl = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(eucl)
