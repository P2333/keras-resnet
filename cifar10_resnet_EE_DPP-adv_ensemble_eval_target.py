from __future__ import print_function
import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
from keras.datasets import cifar10
import tensorflow as tf
import cleverhans.attacks as attacks
from utils_model_eval import model_eval_targetacc
import os
from utils import *
import numpy as np
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
y_test_target = np.zeros_like(y_test)
for i in range(y_test.shape[0]):
    l = np.random.randint(10)
    while l == y_test[i][0]:
        l = np.random.randint(10)
    y_test_target[i][0] = l
print('Finish crafting y_test_target!!!!!!!!!!!')
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
y_test_target = keras.utils.to_categorical(y_test_target, num_classes)





# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
y_target = tf.placeholder(tf.float32, shape=(None, num_classes))

sess = tf.Session()
keras.backend.set_session(sess)



# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(os.getcwd(), 'EE_DPP_saved_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda)+'_'+str(FLAGS.augmentation))
model_name = 'cifar10_%s_model.%d.h5' % (model_type, FLAGS.epoch)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s'% filepath)

# Prepare baseline model pre-trained checkpoints directory.
save_dir_baseline = os.path.join(os.getcwd(), 'EE_DPP_saved_models'+str(FLAGS.num_models)+'_lamda0.0_logdetlamda0.0_'+str(FLAGS.augmentation))
model_name_baseline = 'cifar10_%s_model.%d.h5' % (model_type, FLAGS.baseline_epoch)
filepath_baseline = os.path.join(save_dir_baseline, model_name_baseline)
print('Restore baseline model checkpoints from %s'% filepath_baseline)





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
model = Model(inputs=model_input, outputs=model_output)
model_ensemble = keras.layers.Add()(model_out)
model_ensemble = Model(input=model_input, output=model_ensemble)


#Creat baseline model
model_input_baseline = Input(shape=input_shape)
model_dic_baseline = {}
model_out_baseline = []
if version == 2:
    for i in range(FLAGS.num_models):
        model_dic_baseline[str(i)] = resnet_v2(input=model_input_baseline, depth=depth)
        model_out_baseline.append(model_dic_baseline[str(i)][2])
else:
    for i in range(FLAGS.num_models):
        model_dic_baseline[str(i)] = resnet_v1(input=model_input_baseline, depth=depth)
        model_out_baseline.append(model_dic_baseline[str(i)][2])
model_output_baseline = keras.layers.concatenate(model_out_baseline)
model_baseline = Model(inputs=model_input_baseline, outputs=model_output_baseline)
model_ensemble_baseline = keras.layers.Add()(model_out_baseline)
model_ensemble_baseline = Model(input=model_input_baseline, output=model_ensemble_baseline)



#Get individual models
wrap_ensemble = KerasModelWrapper(model_ensemble)
wrap_ensemble_baseline = KerasModelWrapper(model_ensemble_baseline)



#Load model
model.load_weights(filepath)
model_baseline.load_weights(filepath_baseline)



# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap_ensemble)
    att_baseline = attacks.MadryEtAl(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.MadryEtAl(wrap_ensemble)
    att_baseline = attacks.MadryEtAl(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att = attacks.MadryEtAl(wrap_ensemble)
    att_baseline = attacks.MadryEtAl(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'BasicIterativeMethod':
    att = attacks.BasicIterativeMethod(wrap_ensemble)
    att_baseline = attacks.BasicIterativeMethod(wrap_ensemble_baseline)

# Consider the attack to be constant
success_rate = np.zeros((2,11)) #first row is our method, second
eval_par = {'batch_size': 500}

for eps in range(11):
    eps_ = eps * 0.01
    print('eps is %.2f'%eps_)
    att_params = {'eps': eps_,
                      'clip_min': clip_min,
                      'clip_max': clip_max,
                      'nb_iter': 10,
                      'y_target': y_target}
    adv_x = tf.stop_gradient(att.generate(x, **att_params))
    adv_x_baseline = tf.stop_gradient(att_baseline.generate(x, **att_params))
    preds = model_ensemble(adv_x)
    preds_baseline = model_ensemble_baseline(adv_x_baseline)
    acc = model_eval_targetacc(sess, x, y, y_target, preds, x_test, y_test_target, args=eval_par)
    acc_baseline = model_eval_targetacc(sess, x, y, y_target, preds_baseline, x_test, y_test_target, args=eval_par)
    success_rate[0][eps] = acc
    success_rate[1][eps] = acc_baseline
    print('adv_ensemble_acc_target: %.3f adv_ensemble_baseline_acc_target: %.3f' % (acc, acc_baseline))

np.savetxt('output_results/cifar10_adv_ensemble_acc_models' + str(FLAGS.num_models) + '_lamda' + str(FLAGS.lamda) + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_' + FLAGS.attack_method + '_target.txt', success_rate)