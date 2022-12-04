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

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, UpSampling2D, Reshape


path_img = "./demo/CelebA_subset/images"
csv_path = './demo/CelebA_subset/list_attr_celeba.csv'


class DatasetCelebA():

    # sur cette classe on assurent les transformations faite sur la DATA
    # (taille des images 256,256)
    # deux attributs dataset(les images) et attr()

    def __init__(self, root, attr):
        self.dataset = tfds.folder_dataset.ImageFolder(
            root_dir=root,
            shape=(256, 256, 3)
        )

        self.attr = pd.read_csv(attr)

    def __len__(self):
        return len(self.dataset)


dataset = DatasetCelebA(root=path_img, attr=csv_path)

NomImg = np.sort(os.listdir(path_img))

print(len(NomImg))
lenTest, lenTrain = int(len(NomImg) - (len(NomImg) * 0.8)), int(
    len(NomImg) - (len(
        NomImg) * 0.2))

NomImgTrain = NomImg[:lenTrain]
NomImgTest = NomImg[lenTrain:lenTrain + lenTest]
NomImgAll = NomImg[:]


def DataTrain(X):
    XorY = []
    for i, e in enumerate(X):
        image = load_img(path_img + "/" + e, target_size=(256, 256, 3))
        image = img_to_array(image) / 255.0
        XorY.append(image)
    XorY = np.array(XorY)
    return (XorY)


X_train = DataTrain(NomImgTrain)
print("X_train.shape = {}".format(X_train.shape))
X_test = DataTrain(NomImgTest)
print("X_test.shape = {}".format(X_test.shape))
# X_all =(tf.data.Dataset.from_tensor_slices(DataTrain(NomImgAll)))
X_all = tf.convert_to_tensor(DataTrain(NomImgAll))


# print("data_all.shape = {}".format(X_all.shape))

