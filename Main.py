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






