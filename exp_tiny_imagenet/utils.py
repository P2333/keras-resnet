from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

# Training parameters
num_classes = 200
log_offset = 1e-20
det_offset = 1e-5

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lamda', 1.0, "lamda for Ensemble Entropy(EE)")
tf.app.flags.DEFINE_float('log_det_lamda', 0.5, "lamda for non-ME")

tf.app.flags.DEFINE_integer('num_models', 3, "The num of models in the ensemble")
tf.app.flags.DEFINE_integer('epoch', 1, "epoch of the checkpoint to load")

tf.app.flags.DEFINE_string('attack_method', 'MadryEtAl', "FastGradientMethod, MadryEtAl")
tf.app.flags.DEFINE_float('eps', 0.05, "maximal eps for attacks")
tf.app.flags.DEFINE_integer('baseline_epoch', 1, "epoch of the checkpoint to load")

tf.app.flags.DEFINE_string('tiny_imagenet_train_dir', '/home/tianyu/adv_train/imagenet_data/train', '')
tf.app.flags.DEFINE_string('tiny_imagenet_validation_dir', '/home/tianyu/adv_train/imagenet_data/validation', '')
tf.app.flags.DEFINE_integer('train_batch_size', 64, '')
tf.app.flags.DEFINE_integer('validation_batch_size', 64, '')

zero = tf.constant(0, dtype=tf.float32)

## Functions ##
def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)

def Ensemble_Entropy(y_true, y_pred, num_model=FLAGS.num_models, batch_size=FLAGS.train_batch_size):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble

def log_det(y_true, y_pred, num_model=FLAGS.num_models, batch_size=FLAGS.train_batch_size):
    bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred + det_offset
    mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det

## Metrics ##
def Ensemble_Entropy_metric(y_true, y_pred, num_model=FLAGS.num_models, batch_size=FLAGS.train_batch_size):
    EE = Ensemble_Entropy(y_true, y_pred, num_model=num_model, batch_size=batch_size)
    return K.mean(EE)

def log_det_metric(y_true, y_pred, num_model=FLAGS.num_models, batch_size=FLAGS.train_batch_size):
    log_dets = log_det(y_true, y_pred, num_model=num_model, batch_size=batch_size)
    return K.mean(log_dets)

def acc_metric(y_true, y_pred, num_model=FLAGS.num_models, batch_size=FLAGS.train_batch_size):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    acc = 0
    for i in range(num_model):
        acc += keras.metrics.categorical_accuracy(y_t[i], y_p[i])
    return acc / num_model

def Loss_withEE_DPP(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    log_dets = log_det(y_true, y_pred, num_model, FLAGS.train_batch_size)
    return CE_all - FLAGS.lamda * EE - FLAGS.log_det_lamda * log_dets