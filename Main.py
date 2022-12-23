#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:59:23 2022

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
#from data_attributes import create_attributes, switch_att,switch_att1
#from train import train
import glob
import cv2 as cv

#from tensorflow.keras.utils import Progbar
import time as time


autoencoder = model.Autoencoder()
batch_size = 32
path_img = '/content/img_align_celeba'

csv_path = '/content/gdrive/MyDrive/list_attr_celeba.csv'
Data=data.DatasetCelebA(path_img,csv_path,min_data=0,max_data=3000,load=True)


epoch_size = 2400
eval_size = 600
max_epoch = 100
autoencoder.compile(tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5),tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5))
""""
def tilles(batch):
    attributes=[]
    for i in batch:
        a=np.tile(np.concatenate((i[:,0],i[:,1])),(4,1))
        features=tf.reshape(a,(2,2,80))
        attributes.append(features)
    return attributes"""

def convert_time(total_s):
	total_s = np.round(total_s / 60) * 60
	time_h = np.floor(total_s / 3600)
	total_s -= time_h * 3600
	time_m = np.floor(total_s / 60)
	total_s -= time_m * 60
	time_s = total_s
	return '%ih:%im:%.2fs'%(time_h,time_m,time_s)

average_time_step = []
total_steps = max_epoch*epoch_size
AE_loss=[]
D_loss=[]
AE_val_Loss=[]
D_val_loss=[]
for epoch in range(max_epoch):
  print(epoch)
  start_epoch = time.time()
  #train_progbar = Progbar(epoch_size)
  #print("train_progbar",train_progbar)
  #eval_progbar = Progbar(eval_size)
  #Training
  #print(f"Epoch {history.epoch} / {max_epoch}")
  start_step = time.time()
  for step in range(0, epoch_size, batch_size):
    if step>=500:
      l=0.0001
    else:
      l=0.0
    
    x,attr= Data.batch_train(batch_size)
   

    
    dloss,modloss = autoencoder.train_step(autoencoder,x,attr,l)
    
    #train_progbar.add(batch_size, values = [dloss,modloss])
   
  
  for step in range(0,eval_size,batch_size):

    if step>=100:
      l=0.0001
    else:
      l=0.0
    x,att = Data.eval_batch(batch_size, Data.train_indice, Data.val_indice)
    #features=tilles(features)
    dis_val_loss, AE_val_loss= autoencoder.test_step(autoencoder,x,att,l)

    #eval_progbar.add(batch_size, values = dic_to_tab(metrics))
    #history.update(metrics)
  #
  average_time_step.append(time.time()-start_step)
  print(f"\rEpoch %i/%i - Step %i/%i - Loss_AE :{modloss.numpy()} - loss_disc:{dloss.numpy()} - AE_val_loss:{AE_val_loss.numpy()} - dis_val_loss : %s%.3f (remaining time : %s)" 
				%(epoch+1, max_epoch, epoch_size,epoch_size , ' '*(4-len(str(int(np.round(modloss.numpy()))))),dis_val_loss.numpy(), convert_time(
					total_s=(total_steps - epoch*epoch_size - step) * np.mean(average_time_step))), end='')
  # Je prefere afficher a chaque epoch
  """if epoch % 1 == 0:
    if epoch%5 == 0:
      autoencoder.save_weights(f"models/fadernetwork/{epoch}/weights")"""
  AE_loss.append(modloss.numpy())
  D_loss.append(dloss.numpy())
  AE_val_Loss.append(AE_val_loss.numpy())
  D_val_loss.append(dis_val_loss.numpy()) 
autoencoder.save("final/weights",save_format='tf')
print("\n",AE_loss)
print("\n",AE_val_Loss)
print("\n",D_loss)
print("\n",D_val_loss)







