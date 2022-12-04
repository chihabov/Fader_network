#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 21:27:09 2022

@author: spi-2019-39
"""
from natsort import natsorted, ns
import csv
import os
from abc import ABC
import keras
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Model
import model
import numpy as np
from keras import layers, models
from tensorflow.keras.optimizers import Adam
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds
import data
import matplotlib.pyplot as plt
from data_attributes import create_attributes, switch_att
from data_attributes import attr_ind

"""
def train(data_batch,attr_batch,attr_ind,AE, AE_optimizer,num_epochs=10,l=0, dldx=2e-10,AE_MSE=tf.keras.losses.MeanSquaredError()):

    reconstruction_losses = []
    AE_losses = []
    D_losses = []
    rec_inv_tradeoff = l

    for epoch in range(num_epochs):
        print(epoch)
        for i in range(data_batch.shape[0]):
            img=data_batch[i]
            
            
            
            young_blond = tf.stack((attr_ind(39, data_batch[i]), attr_ind(9, data_batch[i])),
                                   axis=1)
            male_attractive = tf.stack((attr_ind(20, data_batch[i]), attr_ind(2, data_batch[i])), axis=1)
            features =tf.reshape(tf.stack((young_blond, male_attractive), axis=2),(-1,2,2,1))
            features= tf.cast(features, tf.float32)
            #Y =tf.reshape( tf.stack((young_blond, male_attractive), axis=2),(-1,4))

            reconstructed = AE(img)




            # loss MSE pour l'autoencodeur
            reconstruction_loss = AE_MSE(reconstructed, img)
            
            
            
        reconstruction_losses.append(reconstruction_loss)#[0])
        #AE_scheduler.step(AE_loss.data[0])

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title(r'$\mathcal{L}_{reconstruction}$', fontsize=20)
        plt.plot(reconstruction_losses)
        plt.grid()
        





    return reconstruction_losses
"""