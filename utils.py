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
tf.app.flags.DEFINE_float('lamda', 1.0, "lamda for Ensemble Entropy(EE)")
tf.app.flags.DEFINE_float('nonME_lamda', 0.5, "lamda for non-ME")
tf.app.flags.DEFINE_bool('augmentation', False, "whether use data augmentation")
tf.app.flags.DEFINE_integer('num_models', 2, "The num of models in the ensemble")
tf.app.flags.DEFINE_integer('epoch', 1, "epoch of the checkpoint to load")

def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)

def Ensemble_Entropy(y_true, y_pred, num_model=FLAGS.num_models, batch_size=batch_size):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble

def non_ME (y_true, y_pred, num_model=FLAGS.num_models, batch_size=batch_size):
    ONE = tf.ones_like(y_true) # batch_size X (num_class X num_models)
    R_y_true = ONE - y_true
    non_y_pred = tf.multiply(R_y_true, y_pred)
    y_p = tf.split(non_y_pred, num_model, axis=-1)
    sum_nonme = 0
    for i in range(num_model):
        sum_nonme += Entropy(y_p[i] / tf.reduce_sum(y_p[i], axis=-1, keep_dims=True))
    return sum_nonme

def Ensemble_Entropy_metric(y_true, y_pred, num_model=FLAGS.num_models, batch_size=batch_size):
    EE = Ensemble_Entropy(y_true, y_pred, num_model=num_model, batch_size=batch_size)
    return K.mean(EE)

def non_ME_metric (y_true, y_pred, num_model=FLAGS.num_models, batch_size=batch_size):
    sum_nonme = non_ME(y_true, y_pred, num_model, batch_size)
    return sum_nonme / num_model

def acc_metric(y_true, y_pred, num_model=FLAGS.num_models, batch_size=batch_size):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    acc = 0
    for i in range(num_model):
        acc += keras.metrics.categorical_accuracy(y_t[i], y_p[i])
    return acc / num_model

def Loss_withEE_plus(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    sum_nonme = non_ME(y_true, y_pred, num_model)
    return CE_all - FLAGS.lamda * EE + FLAGS.nonME_lamda * sum_nonme