"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
from keras.datasets import cifar10
from keras import backend
import tensorflow as tf
import numpy as np
import os
from model import resnet_v1, resnet_v2
from utils import *
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_eval


# Training parameters
num_classes = 10
log_offset = 1e-20
n = 3
subtract_pixel_mean = True# Subtracting pixel mean improves accuracy



# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print(model_type)


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
clip_min = 0.0
clip_max = 1.0
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)





# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
sess = tf.Session()
keras.backend.set_session(sess)



# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(os.getcwd(), 'EEplus_saved_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_nonMElamda'+str(FLAGS.nonME_lamda)+'_'+str(FLAGS.augmentation))
model_name = 'cifar10_%s_model.%d.h5' % (model_type, FLAGS.epoch)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s'% filepath)

#Creat model
model_input = Input(shape=input_shape)
model_dic = {}
model_out = []
if version == 2:
    for i in range(FLAGS.num_models):
        model_dic[str(i)] = resnet_v2(input=model_input, depth=depth)
        model_out.append(model_dic[str(i)][2])
else:
    for i in range(FLAGS.num_models):
        model_dic[str(i)] = resnet_v1(input=model_input, depth=depth)
        model_out.append(model_dic[str(i)][2])
model_output = keras.layers.concatenate(model_out)
model = Model(input=model_input, output=model_output)

#Wrap models
model_0 = model_dic['0'][0]
model_1 = model_dic['2'][0]
wrap = KerasModelWrapper(model_1)


#Load model
model.load_weights(filepath)


# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
fgsm = FastGradientMethod(wrap)
fgsm_params = {'eps': 0.05,
               'clip_min': clip_min,
               'clip_max': clip_max}
adv_x = fgsm.generate(x, **fgsm_params)


# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv_transfer = model_0(adv_x)
preds_adv = model_1(adv_x)
preds = model_0(x)


# Evaluate the accuracy of the MNIST model on adversarial examples
eval_par = {'batch_size': 100}
acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
acc_adv = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
acc_adv_transfer = model_eval(sess, x, y, preds_adv_transfer, x_test, y_test, args=eval_par)

print(acc)
print(acc_adv)
print(acc_adv_transfer)


