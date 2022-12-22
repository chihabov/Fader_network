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


  def __init__(self,encoder, decoder, discriminator):
    super(Autoencoder, self).__init__()
    self.encod=encoder
    self.decod=decoder
    self.disc=discriminator
  def compile(self, disc_opti, auto_opti):
        super(Autoencoder, self).compile()
        self.d_opti = disc_opti
        self.Disc_loss = keras.losses.BinaryCrossentropy(from_logits = True)
        self.AE_loss = keras.losses.MeanSquaredError()
        self.g_opti = auto_opti
        self.disc.compile(
            self.d_opti,
            self.Disc_loss)
        self.decod.compile(
            self.g_opti,
            self.AE_loss)
        
  def train_step(self,data_batch,attr_batch,features,l=0):
            self.data_batch=data_batch
            self.attr_batch=attr_batch
            self.l=l
            self.features=features
        
            img=self.data_batch
            img=tf.Variable(img)

            self.disc.trainable = True
            self.decod.trainable = False            
           
            with tf.GradientTape() as Tape:
                    ez=self.encod(img)
                    outdiscriminator = self.disc(ez)
                    #print("taille sortie discriminator",outdiscriminator.shape)
                    #print("real attr_batch",self.features.shape)
                    D_loss=self.Disc_loss(self.attr_batch,outdiscriminator)
                    #print(D_loss)
                    D_loss2=self.Disc_loss(1-self.attr_batch,outdiscriminator)
                    #print(D_loss2)
                    
            gradient_rec = Tape.gradient(D_loss, self.disc.trainable_weights)
            self.d_opti.apply_gradients(zip(gradient_rec, self.disc.trainable_weights))                         
            
            #gradient_diss = Tape.gradient(D_loss, AE.discriminator.trainable_weights)
            #AE.discriminator.optimizer.apply_gradients(zip(gradient_diss, AE.discriminator.trainable_weights))
            self.disc.trainable = False
            self.decod.trainable = True
            self.encod.trainable=True
            #img.features=input
            img=tf.Variable(img)
            with tf.GradientTape() as Tape:
                ez=self.encod(img)
                inp=tf.concat((ez, self.features),axis=3)
                x_reconstruct=self.decod(inp)
                outdiscrim = self.disc(ez)
                flipt_attr = 1 - self.attr_batch
                loss_reconstruct = self.AE_loss(img,x_reconstruct)
                #print("losss_reconstruct",loss_reconstruct)
                loss_model = loss_reconstruct + self.l*self.Disc_loss(flipt_attr, outdiscrim)
    
            gradient_rec = Tape.gradient(loss_model, self.decod.trainable_weights)
            self.g_opti.apply_gradients(zip(gradient_rec, self.decod.trainable_weights))  
            
            return D_loss,  loss_model
  def test_step(self,data_batch,attr_batch,features,l=0):
      
        self.data_batch=data_batch
        self.attr_batch=attr_batch
        self.l=l
        self.features=features
        img=self.data_batch
        img=tf.Variable(img)
        ez=self.encod(img)
        inp=tf.concat((ez, self.features),axis=3)
        x_reconstruct=self.decod(inp)
        outdiscrim = self.disc(ez)
        D_loss=self.Disc_loss(self.attr_batch,outdiscrim)
        flipt_attr = 1 - self.attr_batch
        loss_reconstruct = self.AE_loss(img,x_reconstruct)
        loss_model = loss_reconstruct + self.l*self.Disc_loss(flipt_attr, outdiscrim)
        return D_loss, loss_model



