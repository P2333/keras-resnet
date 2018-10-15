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




#Get individual models
model_0 = model_dic['0'][0] #single model 0
model_1 = model_dic['1'][0] #single model 1
model_2 = model_dic['2'][0] #single model 2
model_3 = Model(inputs=model_input, outputs=keras.layers.Add()(model_out)) #ensemble model
wrap_0 = KerasModelWrapper(model_0)
wrap_1 = KerasModelWrapper(model_1)
wrap_2 = KerasModelWrapper(model_2)
wrap_3 = KerasModelWrapper(model_3)


#Get individual baseline models
model_0_baseline = model_dic_baseline['0'][0]
model_1_baseline = model_dic_baseline['1'][0]
model_2_baseline = model_dic_baseline['2'][0]
model_3_baseline = Model(inputs=model_input_baseline, outputs=keras.layers.Add()(model_out_baseline))
wrap_0_baseline = KerasModelWrapper(model_0_baseline)
wrap_1_baseline = KerasModelWrapper(model_1_baseline)
wrap_2_baseline = KerasModelWrapper(model_2_baseline)
wrap_3_baseline = KerasModelWrapper(model_3_baseline)



#Load model
model.load_weights(filepath)
model_baseline.load_weights(filepath_baseline)



# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att_0 = attacks.MadryEtAl(wrap_0)
    att_1 = attacks.MadryEtAl(wrap_1)
    att_2 = attacks.MadryEtAl(wrap_2)
    att_3 = attacks.MadryEtAl(wrap_3)
    att_0_baseline = attacks.MadryEtAl(wrap_0_baseline)
    att_1_baseline = attacks.MadryEtAl(wrap_1_baseline)
    att_2_baseline = attacks.MadryEtAl(wrap_2_baseline)
    att_3_baseline = attacks.MadryEtAl(wrap_3_baseline)
    att_params = {'eps': FLAGS.eps,
                   'clip_min': clip_min,
                   'clip_max': clip_max,
                   'nb_iter': 10}
elif FLAGS.attack_method == 'FastGradientMethod':
    att_0 = attacks.FastGradientMethod(wrap_0)
    att_1 = attacks.FastGradientMethod(wrap_1)
    att_2 = attacks.FastGradientMethod(wrap_2)
    att_3 = attacks.FastGradientMethod(wrap_3)
    att_0_baseline = attacks.FastGradientMethod(wrap_0_baseline)
    att_1_baseline = attacks.FastGradientMethod(wrap_1_baseline)
    att_2_baseline = attacks.FastGradientMethod(wrap_2_baseline)
    att_3_baseline = attacks.FastGradientMethod(wrap_3_baseline)
    att_params = {'eps': FLAGS.eps,
                  'clip_min': clip_min,
                  'clip_max': clip_max}
elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att_0 = attacks.MomentumIterativeMethod(wrap_0)
    att_1 = attacks.MomentumIterativeMethod(wrap_1)
    att_2 = attacks.MomentumIterativeMethod(wrap_2)
    att_3 = attacks.MomentumIterativeMethod(wrap_3)
    att_0_baseline = attacks.MomentumIterativeMethod(wrap_0_baseline)
    att_1_baseline = attacks.MomentumIterativeMethod(wrap_1_baseline)
    att_2_baseline = attacks.MomentumIterativeMethod(wrap_2_baseline)
    att_3_baseline = attacks.MomentumIterativeMethod(wrap_3_baseline)
    att_params = {'eps': FLAGS.eps,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                  'nb_iter': 10}

adv_x_0 = tf.stop_gradient(att_0.generate(x, **att_params))
adv_x_1 = tf.stop_gradient(att_1.generate(x, **att_params))
adv_x_2 = tf.stop_gradient(att_2.generate(x, **att_params))
adv_x_3 = tf.stop_gradient(att_3.generate(x, **att_params))

adv_x_0_baseline = tf.stop_gradient(att_0_baseline.generate(x, **att_params))
adv_x_1_baseline = tf.stop_gradient(att_1_baseline.generate(x, **att_params))
adv_x_2_baseline = tf.stop_gradient(att_2_baseline.generate(x, **att_params))
adv_x_3_baseline = tf.stop_gradient(att_3_baseline.generate(x, **att_params))


# Consider the attack to be constant
acc_record = np.zeros((8,8))
eval_par = {'batch_size': 500}
for i in range(8):
    for j in range(8):
        index_adv = i%4
        index_model = j%4
        baseline_adv = ''
        baseline_model = ''
        if i>3:
            baseline_adv = '_baseline'
        if j>3:
            baseline_model = '_baseline'
        print('model_'+str(index_adv)+baseline_adv+' transfer to model_'+str(index_model)+baseline_model)
        preds=eval('model_'+str(index_model)+baseline_model+'(adv_x_'+str(index_adv)+baseline_adv+')')
        acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
        acc_record[i][j] = acc
        print('adv_acc is: %.3f'%acc)
np.savetxt('cifar10_transfer_acc_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda)+'_eps'+str(FLAGS.eps)+'_'+FLAGS.attack_method+'_withensemble.txt', acc_record)
