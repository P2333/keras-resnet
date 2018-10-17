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
print('Attack method is %s'%FLAGS.attack_method)

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
if FLAGS.attack_method == 'SaliencyMapMethod':
    att = attacks.SaliencyMapMethod(wrap_ensemble)
    att_baseline = attacks.SaliencyMapMethod(wrap_ensemble_baseline)
    att_params = {'theta': 1., 'gamma': 0.1,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
elif FLAGS.attack_method == 'CarliniWagnerL2':
    att = attacks.CarliniWagnerL2(wrap_ensemble)
    att_baseline = attacks.CarliniWagnerL2(wrap_ensemble_baseline)
    att_params = {'batch_size':500,
                  'confidence': 0.9,
                  'learning_rate': 0.01,
                  'binary_search_steps': 1,
                  'max_iterations': 1000,
                  'initial_const': 0.001,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
elif FLAGS.attack_method == 'DeepFool':
    att = attacks.DeepFool(wrap_ensemble)
    att_baseline = attacks.DeepFool(wrap_ensemble_baseline)
    att_params = {'max_iter': 100,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
elif FLAGS.attack_method == 'LBFGS':
    att = attacks.LBFGS(wrap_ensemble)
    att_baseline = attacks.LBFGS(wrap_ensemble_baseline)
    att_params = {'batch_size':500,
                  'confidence': 0.9,
                  'learning_rate': 0.01,
                  'binary_search_steps': 1,
                  'max_iterations': 1000,
                  'initial_const': 0.001,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
# Consider the attack to be constant
acc_record = np.zeros((2,11)) #first row is our method, second
eval_par = {'batch_size': 500}

for eps in range(11):
    eps_ = eps * 0.01
    print('eps is %.2f'%eps_)
    att_params = {'eps': eps_,
                   'clip_min': clip_min,
                   'clip_max': clip_max,
                   'nb_iter': 10}
    adv_x = tf.stop_gradient(att.generate(x, **att_params))
    adv_x_baseline = tf.stop_gradient(att_baseline.generate(x, **att_params))
    preds = model_ensemble(adv_x)
    preds_baseline = model_ensemble_baseline(adv_x_baseline)
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
    acc_baseline = model_eval(sess, x, y, preds_baseline, x_test, y_test, args=eval_par)
    acc_record[0][eps] = acc
    acc_record[1][eps] = acc_baseline
    print('adv_ensemble_acc: %.3f adv_ensemble_baseline_acc: %.3f'%(acc,acc_baseline))

np.savetxt('output_results/cifar10_adv_ensemble_acc_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda)+'_'+FLAGS.attack_method+'.txt', acc_record)