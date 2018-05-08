# -*- coding: utf-8 -*-
"""DCGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kz8rcTZ-KLnqY8TrAho4D44Z_Szd78rF
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os
os.mkdir('images/')

import numpy as np

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(self.latent_dim,)))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """
        This is the part my 'Capsule Layer as a Discriminator in Generative Adversarial Networks' paper focuses on,
        as it introduces a new structure to the discriminator of DCGAN by using Capsule Layers architecture from original
        'Dynamic Routing Between Capsules' paper by S. Sabour, N. Frosst and G. Hinton.

        Discriminator takes real/generated images and outputs its prediction.
        """

        # depending on dataset we define input shape for our network
        img = Input(shape=self.img_shape)
        
        # first typical convlayer outputs a 20x20x256 matrix
        x = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', name='conv1')(img)
        x = LeakyReLU()(x)

        # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
        x = BatchNormalization(momentum=0.8)(x)


        """
        NOTE: Capsule architecture starts from here.
        """
        #
        # primarycaps coming first
        #

        # filters 256 (n_vectors=8 * channels=32)
        x = Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)

        # reshape into the 8D vector for all 32 feature maps combined
        # (primary capsule has collections of activations which denote orientation of the digit
        # while intensity of the vector which denotes the presence of the digit)
        x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)

        # the purpose is to output a number between 0 and 1 for each capsule where the length of the input decides the amount
        x = Lambda(squash, name='primarycap_squash')(x)
        x = BatchNormalization(momentum=0.8)(x)


        #
        # digitcaps are here
        #
        """
        NOTE: My approach is a simplified version of digitcaps i.e. without expanding dimensions into
        [None, 1, input_n_vectors, input_dim_capsule (feature maps)]
        and tiling it into [None, num_capsule, input_n_vectors, input_dim_capsule (feature maps)].
        Instead I replace it with ordinary Keras Dense layers as weight holders in the following lines.

        ANY CORRECTIONS ARE APPRECIATED IN THIS PART, PLEASE SUBMIT PULL REQUESTS!
        """
        x = Flatten()(x)
        # capsule (i) in a lower-level layer needs to decide how to send its output vector to higher-level capsules (j)
        # it makes this decision by changing scalar weight (c=coupling coefficient) that will multiply its output vector and then be treated as input to a higher-level capsule
        #
        # uhat = prediction vector, w = weight matrix but will act as a dense layer, u = output from a previous layer
        # uhat = u * w
        # neurons 160 (num_capsules=10 * num_vectors=16)
        uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)

        # c = coupling coefficient (softmax over the bias weights, log prior) | "the coupling coefficients between capsule (i) and all the capsules in the layer above sum to 1"
        # we treat the coupling coefficiant as a softmax over bias weights from the previous dense layer
        c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one

        # s_j (output of the current capsule level) = uhat * c
        c = Dense(160)(c) # compute s_j
        x = Multiply()([uhat, c])
        """
        NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
        """
        s_j = LeakyReLU()(x)


        #
        # we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
        #
        c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(160)(c) # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(160)(c) # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        pred = Dense(1, activation='sigmoid')(s_j)


        return Model(img, pred)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

dcgan = DCGAN()
dcgan.train(epochs=4000, batch_size=32, save_interval=50)
