#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:48:02 2022

@author: spi-2019-39
"""
from natsort import natsorted, ns
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

class Autoencoder(Model):
  def __init__(self):
        super().__init__()
        self.embeddings = None
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256, 256, 3)),
      layers.Conv2D(16,kernel_size=4, padding='same', strides=2),
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(32,kernel_size=4, padding='same', strides=2),
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(64,kernel_size=4, padding='same', strides=2),
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(128, kernel_size=4, padding='same', strides=2),
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),      
      layers.Conv2D(256, kernel_size=4, padding='same', strides=2),     
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(512, kernel_size=4, padding='same',strides=2),      
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(512, kernel_size=4, padding='same', strides=2),     
      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(1, kernel_size=4, padding='same')])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(1, kernel_size=4, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(512, kernel_size=4, strides=2,padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      
      layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),      
      layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
      layers.ReLU(),   
      layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2,padding='same')
      ])

  def call(self, x):
    encoder=self.encoder(x)
    decoded = self.decoder(encoder)
    return decoded
#tf.concat((self.embeddings, features),axis=1)
