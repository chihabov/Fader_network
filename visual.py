#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:54:34 2022
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:54:34 2022

@author: spi-2019-39

from natsort import natsorted, ns
import csv
import os
from abc import ABC
import keras
import tensorflow as tf
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
import matplotlib.pyplot as plt
AE = model.Autoencoder(pretrained=True)


path_img = '/content/img_align_celeba'
batch_size = 5
csv_path = '/content/gdrive/MyDrive/list_attr_celeba.csv'
Data=data.DatasetCelebA(path_img,csv_path,min_data=100,max_data=106,load=True)

def change_gender(model, Data, n_images=5, img_size=4, factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    YOUNG_INDEX = 39
    BLOND_INDEX = 9
    ATTRACTIVE_INDEX = 2
    x,attr= Data.batch_train(batch_size)
    young_blond = tf.stack((attr[:, 39], attr[:, 9]), axis=1)

    
    reconstructed = []
    factors_count = len(factors)
    
    for index, factor in enumerate(factors):
        male_attractive = tf.stack((tf.ones([5])*factor, attr[:, 2]), axis=1)
        features = tf.reshape(tf.stack((young_blond, male_attractive),
                                   axis=2), [-1, 2, 2, 1])

        reconstructed.append(AE(x, features).numpy())
    for i in range(n_images):
        plt.figure(figsize=((factors_count + 1) * img_size, n_images * img_size))
        plt.subplot(n_images, factors_count + 1, 1)
        plt.imshow(x[i])
        plt.axis('off')
        
        for j in range(factors_count):
            plt.subplot(n_images, factors_count + 1, j + 2)
            plt.imshow((reconstructed[j][i] * 255).astype(np.uint8))
            plt.axis('off')

    plt.show()



change_gender(AE, Data)




 
"""
import matplotlib.pyplot as plt

def Affichge(x):
 """ show images of the datasets"""
    fig = plt.figure(figsize=(10,10))
    nplot = 10
    for i in range(1, nplot):
        ax = fig.add_subplot(1, nplot, i)
        ax.imshow(x[i])
    plt.show()  
    
    
