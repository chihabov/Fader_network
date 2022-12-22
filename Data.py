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

    def __init__(self,root,attr,validation_split=0.25, test_split=0.25,
                 max_data=None,load=None):
        self.root = root
        self.attr = attr
        self.attributes_path=open(self.attr)
        self.dataset_path=os.listdir(self.root)
        self.dataset_path=natsorted(self.dataset_path, alg=ns.PATH | ns.IGNORECASE)
        if max_data is not None:
            self.dataset_path=self.dataset_path[0:max_data]
            #self.attributes=self.switch_att1(self.create_attributes(max_data))
            #self.features=self.switch_att(self.attributes)
           #self.attr = pd.read_csv(attr)
        self.val_split= validation_split
        self.test_split= test_split
        self.train_split=1-validation_split-test_split
        self.load=load
        if self.load:
            self.images=self.load_all_data()
            #print(len(self.images))
            self.attributes=self.switch_att1(self.create_attributes(len(self.images)))
            #print(self.attributes.shape)
            #self.features=self.switch_att(self.attributes)
            #print(self.features.shape)
        self.train_indice = self.train_split*len(self)
        self.val_indice =  (self.train_split + self.val_split)*len(self)
        self.test_indice = len(self)
        

