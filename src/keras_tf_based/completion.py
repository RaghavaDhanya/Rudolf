from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K


import matplotlib.pyplot as plt

import sys

import numpy as np


gen_js_file = open('generator.json', 'r')
gen_js= gen_js_file.read()
gen_js_file.close()
generator = model_from_json(gen_js)
# load weights into new model
generator.load_weights("generator_weights.hdf5")

print("loaded generator!")

dis_js_file = open('discriminator.json', 'r')
dis_js= dis_js_file.read()
dis_js_file.close()
discriminator = model_from_json(dis_js)
# load weights into new model
discriminator.load_weights("discriminator_weights.hdf5")

print("loaded discriminator!")

generator.trainable = False 
discriminator.trainable = False

contextual_loss = K.reduce_sum(
    K.contrib.layers.flatten(
        K.abs(K.mul(mask, generated) - K.mul(mask, images))), 1)
