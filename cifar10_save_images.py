"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
from keras.datasets import cifar10
from PIL import Image


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i in range(1000):
    print(i)
    img = Image.fromarray(x_test[i])
    img.save('cifar10_test_imgs/img_'+str(i)+'.png')
