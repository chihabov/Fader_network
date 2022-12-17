import csv
import os
from abc import ABC

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.utils import load_img
from keras.utils import img_to_array

import numpy as np
from keras import layers, models
from keras.optimizers import Adam
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds
import model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, UpSampling2D, Reshape
from data_attributes import create_attributes, switch_att


from TraitementData import *

path_img = "./demo/CelebA_subset/images"
csv_path = './demo/CelebA_subset/list_attr_celeba.csv'



dataset = DatasetCelebA(root=path_img, attr=csv_path)


NomImg = np.sort(os.listdir(path_img))

print(len(NomImg))
lenTest, lenTrain = int(len(NomImg) - (len(NomImg) * 0.8)), int(
    len(NomImg) - (len(
        NomImg) * 0.2))

NomImgTrain = NomImg[:lenTrain]
NomImgTest = NomImg[lenTrain:lenTrain + lenTest]
NomImgAll = NomImg[:]


attributes= create_attributes(csv_path,nbr_data)
attr=switch_att(attributes)

# recupértion de 4 attributs pour toute la data
#  example :    img1 : [young , blond,male, attractive ]

      



#### separation on  batch
batch_size = 25

attr_batch = tf.convert_to_tensor([attr[idx: idx+batch_size] for idx in
                                 range(0,len(attr), batch_size)])
#print(attr_batch.shape)

data_batch = tf.convert_to_tensor([X_all[idx: idx+batch_size] for idx in
range(0,len(X_all), batch_size)])

#________________________#
attr = switch_att(create_attributes(csv_path,101))
print(attr.shape)
#---------------------learning-------------------------------#

 #on this part we collect 4 attributes then we concatenate them in order to have a matrix

young_blond = tf.stack(
    (attr[:,39], attr[:,9]), axis=1)
male_attractive = tf.stack(
    (attr[:,20], attr[:,2]), axis=1)

young_blond_batch = [young_blond[idx: idx + batch_size] for idx in range(0,
                                                                    len(young_blond),
batch_size)]

male_attractive_batch = [male_attractive[idx: idx + batch_size] for idx in
                         range(0, len(male_attractive),
batch_size)]

features = tf.stack((young_blond, male_attractive),
                       axis=2)
true_labels = tf.reshape(tf.stack((young_blond, male_attractive),
                          axis=1),[100,4])








X_train = DataTrain(NomImgTrain,path_img)
print("X_train.shape = {}".format(X_train.shape))
X_test = DataTrain(NomImgTest,path_img)
print("X_test.shape = {}".format(X_test.shape))
# X_all =(tf.data.Dataset.from_tensor_slices(DataTrain(NomImgAll)))
X_all = tf.convert_to_tensor(DataTrain(NomImgAll,path_img))


autoencoder = model.Autoencoder()
autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
autoencoder.fit(X_all,X_all,epochs=20)
autoencoder.encoder.summary()
