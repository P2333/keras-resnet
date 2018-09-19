from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
num_classes = 10
log_offset = 1e-20

FLAGS = tf.app.flags.FLAGS

def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)

def Ensemble_Entropy(y_true, y_pred, num_model=2, batch_size=batch_size):
    y_p_1, y_p_2 = tf.split(y_pred, num_model, axis=-1)
    y_t_1, y_t_2 = tf.split(y_true, num_model, axis=-1)
    Ensemble = Entropy((y_p_1 + y_p_2) / num_model)
    return Ensemble

def non_ME (y_true, y_pred, num_model=2, batch_size=batch_size):
    ONE = tf.ones_like(y_true) # batch_size X (num_class X num_models)
    R_y_true = ONE - y_true
    non_y_pred = tf.multiply(R_y_true, y_pred)
    y_p_1, y_p_2 = tf.split(non_y_pred, num_model, axis=-1)
    non_me1 = Entropy(y_p_1 / tf.reduce_sum(y_p_1, axis=-1, keep_dims=True))
    non_me2 = Entropy(y_p_2 / tf.reduce_sum(y_p_2, axis=-1, keep_dims=True))
    return non_me1, non_me2

def Ensemble_Entropy_metric(y_true, y_pred, num_model=2, batch_size=batch_size):
    EE = Ensemble_Entropy(y_true, y_pred, num_model=num_model, batch_size=batch_size)
    return K.mean(EE)

def non_ME_metric (y_true, y_pred, num_model=2, batch_size=batch_size):
    non_me1, non_me2 = non_ME(y_true, y_pred, num_model, batch_size)
    return (non_me1 + non_me2)/2

def acc_metric(y_true, y_pred, num_model=2, batch_size=batch_size):
    y_p_1, y_p_2 = tf.split(y_pred, num_model, axis=-1)
    y_t_1, y_t_2 = tf.split(y_true, num_model, axis=-1)
    acc1 = keras.metrics.categorical_accuracy(y_t_1, y_p_1)
    acc2 = keras.metrics.categorical_accuracy(y_t_2, y_p_2)
    return (acc1 + acc2)/2

def Loss_withEE_plus(y_true, y_pred, num_model=2):
    y_p_1, y_p_2 = tf.split(y_pred, num_model, axis=-1)
    y_t_1, y_t_2 = tf.split(y_true, num_model, axis=-1)
    CE_1 = keras.losses.categorical_crossentropy(y_t_1, y_p_1)
    CE_2 = keras.losses.categorical_crossentropy(y_t_2, y_p_2)
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    non_me1, non_me2 = non_ME(y_true, y_pred, num_model)
    return CE_1 + CE_2 - FLAGS.lamda * EE + FLAGS.nonME_lamda * (non_me1 + non_me2)