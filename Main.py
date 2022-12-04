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
