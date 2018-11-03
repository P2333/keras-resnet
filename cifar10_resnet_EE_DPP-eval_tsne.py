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
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import os
from model import resnet_v1, resnet_v2
from utils import *


# Training parameters
num_classes = 10
log_offset = 1e-20


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print(model_type)

# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(os.getcwd(), 'EE_DPP_saved_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda)+'_'+str(FLAGS.augmentation))
model_name = 'cifar10_%s_model.%d.h5' % (model_type, FLAGS.epoch)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s'% filepath)

#Creat model
model_input = Input(shape=input_shape)
model_dic = {}
model_out = []
model_features = {}
if version == 2:
    for i in range(FLAGS.num_models):
        model_dic[str(i)] = resnet_v2(input=model_input, depth=depth)
        model_out.append(model_dic[str(i)][2])
        model_features[str(i)] = Model(inputs=model_input, outputs=model_dic[str(i)][4])
else:
    for i in range(FLAGS.num_models):
        model_dic[str(i)] = resnet_v1(input=model_input, depth=depth)
        model_out.append(model_dic[str(i)][2])
        model_features[str(i)] = Model(inputs=model_input, outputs=model_dic[str(i)][4])
model_output = keras.layers.concatenate(model_out)
model = Model(inputs=model_input, outputs=model_output)


#Load model
model.load_weights(filepath)


for i in range(FLAGS.num_models):
    predictions = model_features[str(i)].predict(x_test)
    print(i)
    print(predictions.shape)
    save_results_dir = os.path.join(os.getcwd(), 'output_results/tsne_features')
    if not os.path.isdir(save_results_dir):
        os.makedirs(save_results_dir)
    np.savetxt(os.path.join(save_results_dir, 'tsnefeatures_models%sof%s_lamda%s_logdetlamda%s_epoch%d.txt'%(str(i+1), str(FLAGS.num_models),str(FLAGS.lamda),str(FLAGS.log_det_lamda),FLAGS.epoch)), predictions)
    np.savetxt(os.path.join(save_results_dir, 'test_labels.txt'), y_test)