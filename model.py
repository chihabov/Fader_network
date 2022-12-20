
import csv
import os
from abc import ABC
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Model

import numpy as np
from keras import layers, models
from tensorflow.keras.optimizers import Adam
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds




class Discreminateur(keras.Model):

    def __init__(self):
        super(Discreminateur, self).__init__()
        self.outD = None

        self.discriminator = tf.keras.Sequential([
            layers.Input(shape=(2, 2, 512)),
            layers.Conv2DTranspose(512, kernel_size=4, strides=1,padding='same'),
            layers.Flatten(),
            layers.Dropout(0.3),
            keras.layers.Dense(256, input_shape=(512,), activation=None),
            keras.layers.Dense(4,input_shape=(512,)),
            ])


    def compile(self, disc_opti,d_loss =keras.losses.BinaryCrossentropy(from_logits=True)):
        self.d_opti = disc_opti
        self.Disc_loss = d_loss

        self.discriminator.compile(self.d_opti,self.Disc_loss)
    """def build(self,learning_rate=0.002,beta_1=0.5,
                                        name='Adam'):
           self.lr=learning_rate
           self.b1=beta_1
           self.name=name
           self.discriminator.build(self.lr,self.b1,self.name)"""

    def call(self, embedding):

        self.outD =tf.sigmoid(self.discriminator(embedding))
        return  self.outD
      
      

class Autoencoder(keras.Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.embedding = None
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(16, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(32, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, kernel_size=4, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(2,2,512+1)),
            layers.Conv2DTranspose(512+1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(256 + 1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(128 + 1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(64 + 1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(32 + 1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(16 + 1, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(3, kernel_size=4, strides=2,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])


    def compile(self,auto_opti,AE_l = keras.losses.MeanSquaredError()):

        self.AE_loss = AE_l
        self.g_opti = auto_opti
        self.encoder.compile(
            self.g_opti,
            self.AE_loss)

    """def build(self,learning_rate=0.002,beta_1=0.5,
                                        name='Adam'):
           self.lr=learning_rate
           self.b1=beta_1
           self.name=name
           self.encoder.build(self.lr,self.b1,self.name)"""


    def call(self, input):
        x, features = input
        embedding = self.encoder(x)
        decoded = self.decoder(tf.concat((embedding, features),
                                         axis=3))
        return  decoded,embedding



