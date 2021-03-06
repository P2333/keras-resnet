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
from keras_wraper_ensemble import KerasModelWrapper
from jsma import jsma_impl_loop

# Training parameters
num_classes = 10
log_offset = 1e-20
n = 3
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy

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
print('Attack method is %s' % FLAGS.attack_method)

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
y_adv = tf.placeholder(tf.float32, shape=(None, num_classes))
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
keras.backend.set_session(sess)

if FLAGS.label_smooth == 1.0:
    ls = ''
else:
    ls = '_labelsmooth'+str(FLAGS.label_smooth)

# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(
    '/mfs/tianyu/improve_diversity/keras-resnet',
    'EE_DPP_saved_models' + str(FLAGS.num_models) + '_lamda' + str(FLAGS.lamda)
    + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_' + str(FLAGS.augmentation)+ls)
model_name = 'cifar10_%s_model.%d.h5' % (model_type, FLAGS.epoch)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s' % filepath)

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
model_ensemble = keras.layers.Average()(model_out)
model_ensemble = Model(inputs=model_input, outputs=model_ensemble)

#Get individual models
wrap_ensemble = KerasModelWrapper(model_ensemble)
# for layer in wrap_ensemble.model.layers:
#     print(layer.get_config()['name'])

# wrap_ensemble = KerasModelWrapper(model_dic['1'][0])

#Load model

# adv_x = jsma_attack(
#     wrap_ensemble,
#     x,
#     y_adv,
#     epochs=0.1,
#     eps=1.0,
#     clip_min=clip_min,
#     clip_max=clip_max,
# )
# Consider the attack to be constant
eval_par = {'batch_size': 1}

sess.run(tf.initialize_all_variables())
model.load_weights(filepath)

print("start attack")
# print(adv_x)
# print(y_test[0:2])
# adv_np = sess.run(adv_x, feed_dict={x: x_test[0:1], y_adv: y_test[1:2]})
# print(adv_np)
num_samples = 100
preds = wrap_ensemble.get_probs(x)

print(
    model_eval(
        sess,
        x,
        y,
        preds,
        x_test[:num_samples],
        y_test[:num_samples],
        args=eval_par))

adv_data = np.zeros(x_test.shape)
for i in range(num_samples):
    print(i)
    tmp = jsma_impl_loop(
        sess,
        x_test[i:i + 1],
        y_test[i:i + 1],
        wrap_ensemble,
        x,
        y,
        gamma=0.05,
        eps=0.1,
        clip_min=clip_min,
        clip_max=clip_max,
        increase=False)
    adv_data[i] = tmp[0]

print(
    model_eval(
        sess,
        x,
        y,
        preds,
        adv_data[:num_samples],
        y_test[:num_samples],
        args=eval_par))
print("finished")
