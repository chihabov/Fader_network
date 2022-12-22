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
