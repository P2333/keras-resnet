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
import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import os
from utils import *
from model import resnet_v1, resnet_v2
from cleverhans.utils_keras import KerasModelWrapper

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
print('Attack method is %s; eps is %.3f'%(FLAGS.attack_method,FLAGS.eps))

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


#Get individual models
model_0 = model_dic['0'][0]
model_1 = model_dic['1'][0]

#Create ensemble model
model_ensemble = keras.layers.Add()(model_out)
model_ensemble = Model(input=model_input, output=model_ensemble)

#Wrap models
wrap = KerasModelWrapper(model_1)
wrap_ensemble = KerasModelWrapper(model_ensemble)


#Load model
model.load_weights(filepath)


# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap)
    att_ensemble = attacks.MadryEtAl(wrap_ensemble)
    att_params = {'eps': FLAGS.eps,
                   'clip_min': clip_min,
                   'clip_max': clip_max,
                   'nb_iter': 10}
elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.FastGradientMethod(wrap)
    att_ensemble = attacks.FastGradientMethod(wrap_ensemble)
    att_params = {'eps': FLAGS.eps,
                  'clip_min': clip_min,
                  'clip_max': clip_max}

adv_x = att.generate(x, **att_params)
adv_x_ensemble = att_ensemble.generate(x, **att_params)

# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
adv_x_ensemble = tf.stop_gradient(adv_x_ensemble)

preds_adv_transfer = model_0(adv_x)
preds_adv_ensemble = model_ensemble(adv_x_ensemble)
preds_adv = model_1(adv_x)
preds_ensemble = model_ensemble(x)
preds = model_0(x)


# Evaluate the accuracy of the MNIST model on adversarial examples
eval_par = {'batch_size': 100}
acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
acc_ensemble = model_eval(sess, x, y, preds_ensemble, x_test, y_test, args=eval_par)
acc_adv = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
acc_adv_transfer = model_eval(sess, x, y, preds_adv_transfer, x_test, y_test, args=eval_par)
acc_adv_ensemble = model_eval(sess, x, y, preds_adv_ensemble, x_test, y_test, args=eval_par)

print('model_0: %.2f; model_ensemble: %.2f; adv_model_0: %.2f; adv_transfer_model0to1: %.2f; adv_model_ensemble: %.2f'
      %(acc, acc_ensemble,acc_adv, acc_adv_transfer, acc_adv_ensemble))
