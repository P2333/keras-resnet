from __future__ import print_function
import keras
from keras import optimizers
from keras_imagenet_models.resnet50 import ResNet50
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from imagenet_keras_multi_generator import ImageDataGenerator
from keras.models import Model
from keras_applications import imagenet_utils
import tensorflow as tf
import numpy as np
import os
import math
from imagenet_utils_EE_DPP import *


image_size = 224
epochs = 100

def lr_schedule(epoch):
    lr = 1e-1 * (0.94**math.floor(epoch/2))
    print('Learning rate: ', lr)
    return lr


model_input = Input(shape=(image_size,image_size,3))

model_dic = {}
model_out = []
for i in range(FLAGS.num_models):
    model_dic[str(i)] = ResNet50(weights=None,input_tensor=model_input, model_index=i)
    model_out.append(model_dic[str(i)].output)

model_output = keras.layers.concatenate(model_out)

base_model = Model(input=model_input, output=model_output)


model = keras.utils.multi_gpu_model(base_model, gpus=8, cpu_merge=True, cpu_relocation=False)

model.compile(optimizer=optimizers.SGD(lr=lr_schedule(0), momentum=0.9),
              loss=Loss_withEE_DPP,
              metrics=[acc_metric, Ensemble_Entropy_metric, log_det_metric])

model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'imagenet_resnet50_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda))
#save_dir = os.path.join(os.getcwd(), 'imagenet_resnet50_single_sgd0p1')
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_acc_metric', mode='max', verbose=2, save_best_only=True)


#lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(monitor='acc_metric', factor=0.1, mode=max,
                              patience=1, min_lr=0.001, min_delta=0.001)

#callbacks = [lr_reducer, lr_scheduler]
callbacks = [checkpoint, lr_reducer]

print('Using real-time data augmentation.')
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True,
        preprocessing_function=imagenet_utils.preprocess_input)

validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=imagenet_utils.preprocess_input)


train_generator = train_datagen.flow_from_directory(
        FLAGS.imagenet_train_dir,
        target_size=(image_size, image_size),
        batch_size=FLAGS.train_batch_size,
        class_mode='categorical',
        shuffle=True,
        num_models=FLAGS.num_models)

validation_generator = validation_datagen.flow_from_directory(
        FLAGS.imagenet_validation_dir,
        target_size=(image_size, image_size),
        batch_size=FLAGS.validation_batch_size,
        class_mode='categorical',
        shuffle=False,
        num_models=FLAGS.num_models)


history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=100,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      callbacks=callbacks,
      use_multiprocessing=True,
      workers=60)

