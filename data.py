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

    def __init__(self,root,attr,validation_split=0.2, test_split=0.2,min_data=None,max_data=None,load=None):
        self.root = root
        self.attr = attr
        self.attributes_path=open(self.attr)
        
    
        
        self.dataset_path = sorted(glob.glob(self.root + "/*"))

        if max_data is not None:
            self.dataset_path=self.dataset_path[min_data:max_data]
            print(len(self.dataset_path))
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
            self.attributes=self.switch_att1(self.create_attributes(min_data,len(self.images)))
            #print(self.attributes.shape)
            #self.features=self.switch_att(self.attributes)
            #print(self.features.shape)
        self.train_indice = self.train_split*len(self)
        self.val_indice =  (self.train_split + self.val_split)*len(self)
        self.test_indice = len(self)
    
    def load_all_data(self):
        images = []
        print("loading all imagees")
        for i, p in enumerate(self.dataset_path):
          im=cv.imread(p)
         
          img=cv.resize(im, (256,256))
          

          images.append(img/255.0)
          
          if i % 1000 == 0:
            print('\r' + f'{i}/{len(self.dataset_path)}', end ='')
        return np.array(images)  
    
    
    def load_batch(self,indices):
    
     if not self.load:
      x = []
      att=[]
      feat=[]
      for i in indices:
        im = self.load_image(i)
        attri=self.switch_att1(self.create_attributes(i))
        #fit=self.switch_att(self.switch_att1(self.create_attributes(i)))
        att.append(attri)
        x.append(im)
        #feat.append(fit)
     else:
       x = (self.images[indices])/255.0
       att = self.attributes[indices]
       #feat=self.features[indices]
     return tf.constant(x),tf.constant(att,dtype=tf.float32)#tf.constant(
     # feat,dtype=tf.float32)
        
    def batch_train(self,bs, min_ind=0, max_ind=None):
        if max_ind is None:
             max_ind=self.train_indice
        indices = np.random.randint(min_ind, max_ind, bs)


        return self.load_batch(indices)
    
    def eval_batch(self,bs,ind_min, ind_max):
        indices = np.arange(int(ind_min),int(ind_max))        
        return self.load_batch(indices)
   
        
    
        

