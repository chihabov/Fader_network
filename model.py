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

  
 # create a autoencoder
class Autoencoder(Model):


  def __init__(self,encoder, decoder, discriminator):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=([256, 256,3])),
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
      layers.Conv2D(256, kernel_size=4, padding='same', strides=2),      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(512, kernel_size=4, padding='same',strides=2),      layers.BatchNormalization(),
      layers.LeakyReLU(0.2),
      layers.Conv2D(512, kernel_size=4, padding='same', strides=2),      layers.BatchNormalization(),
      layers.LeakyReLU(0.2)])
      #layers.Conv2D(1, kernel_size=4, padding='same')])

    self.decoder = tf.keras.Sequential([
      layers.Input(shape=([2, 2,512+1])),
      layers.Conv2DTranspose(512+1, kernel_size=4,padding='same',strides=2),
      layers.BatchNormalization(),
      layers.ReLU(),
      
      layers.Conv2DTranspose(256+1, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(128+1, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),      
      layers.Conv2DTranspose(64+1, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
      layers.ReLU(),   
      layers.Conv2DTranspose(32+1, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(16+1, kernel_size=4, strides=2, padding='same'),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2,padding='same')
      ])
    if pretrained:
            model = keras.models.load_model('final/weights.h5')
            self.encoder = model.encoder
            self.decoder = model.decoder
    self.discriminator= tf.keras.Sequential([
        layers.Input(shape=([2, 2, 512])),
        layers.Conv2DTranspose(512, kernel_size=4,padding='same'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        keras.layers.Dense(256, input_shape=(512,), activation=None),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.2),
        keras.layers.Dense(4, input_shape=(512,), activation=None),
        tf.keras.layers.Activation('sigmoid')])


  def compile(self, disc_opti, auto_opti):
        super(Autoencoder, self).compile()
        self.d_opti = disc_opti
        self.Disc_loss = keras.losses.BinaryCrossentropy(from_logits = True)
        self.AE_loss = keras.losses.MeanSquaredError()
        self.g_opti = auto_opti
        self.discriminator.compile(
            self.d_opti,
            self.Disc_loss)
        self.decoder.compile(
            self.g_opti,
            self.AE_loss)
        
  def train_step(self,AE,data_batch,attr_batch,l=0):
            self.data_batch=data_batch
            self.attr_batch=attr_batch
            self.l=l
            #self.features=features
            young_blond = tf.stack((attr_batch[:, 39], attr_batch[:, 9]), axis=1)
            male_attractive = tf.stack((attr_batch[:, 20], attr_batch[:, 2]), axis=1)

            features = tf.reshape(tf.stack((young_blond, male_attractive),
                                   axis=2), [-1, 2, 2, 1])

            true_labels = tf.reshape(tf.stack((young_blond, male_attractive),
                                      axis=1), [-1, 4])
            img=self.data_batch
            

            self.discriminator.trainable = True
            AE.trainable = False  
            self.encoder.trainable = True          
           
            with tf.GradientTape() as Tape:
                    ez=self.encoder(img)
                    outdiscriminator = self.discriminator(ez)
                    #print("taille sortie discriminator",outdiscriminator.shape)
                    #print("real attr_batch",self.features.shape)
                    D_loss=self.Disc_loss(true_labels,outdiscriminator)
                    #print(D_loss)
                    D_loss2=self.Disc_loss(1-true_labels,outdiscriminator)
                    #print(D_loss2)
                    
            gradient_rec = Tape.gradient(D_loss, self.discriminator.trainable_weights)
            self.d_opti.apply_gradients(zip(gradient_rec, self.discriminator.trainable_weights))                         
            
            #gradient_diss = Tape.gradient(D_loss, AE.discriminator.trainable_weights)
            #AE.discriminator.optimizer.apply_gradients(zip(gradient_diss, AE.discriminator.trainable_weights))
            self.discriminator.trainable = False
            AE.trainable = True  
            self.encoder.trainable = True 
            #img.features=input
            
            with tf.GradientTape() as Tape:
                
                #inp=tf.concat((ez, self.features),axis=3)
                ez=self.encoder(img)
                x_reconstruct=AE(img, features)
                outdiscrim = self.discriminator(ez)
                flipt_attr = 1 - true_labels
                loss_reconstruct = self.AE_loss(img,x_reconstruct)
                #print("losss_reconstruct",loss_reconstruct)
                loss_model = loss_reconstruct + self.l*self.Disc_loss(flipt_attr, outdiscrim)
    
            gradient_rec = Tape.gradient(loss_model, AE.trainable_weights)
            self.g_opti.apply_gradients(zip(gradient_rec, AE.trainable_weights))  
            
            return D_loss,  loss_model
  def test_step(self,AE,data_batch,attr_batch,l=0):
      
        self.data_batch=data_batch
        self.attr_batch=attr_batch
        self.l=l
        young_blond = tf.stack((attr_batch[:, 39], attr_batch[:, 9]), axis=1)
        male_attractive = tf.stack((attr_batch[:, 20], attr_batch[:, 2]), axis=1)

        features = tf.reshape(tf.stack((young_blond, male_attractive),
                                axis=2), [-1, 2, 2, 1])

        true_labels = tf.reshape(tf.stack((young_blond, male_attractive),
                                  axis=1), [-1, 4])
        img=self.data_batch
        #img=tf.Variable(img)
        ez=self.encoder(img)
        #inp=tf.concat((ez, self.features),axis=3)
        x_reconstruct=AE(img,features)
        outdiscrim = self.discriminator(ez)
        D_loss=self.Disc_loss(true_labels ,outdiscrim)
        flipt_attr = 1 - true_labels 
        loss_reconstruct = self.AE_loss(img,x_reconstruct)
        loss_model = loss_reconstruct + self.l*self.Disc_loss(flipt_attr, outdiscrim)
        return D_loss, loss_model
  def call(self, x,features):#,features
    
    EZZ=self.encoder(x)
    decoded =self.decoder(tf.concat((EZZ, features),axis=3))
   
    return decoded 
