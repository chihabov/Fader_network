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

"""
autoencoder = model.Autoencoder()
autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
autoencoder.fit(X_all,X_all,epochs=20)
autoencoder.encoder.summary()
"""


attr = switch_att(create_attributes(csv_path,101))
print(attr.shape)
#---------------------learning-------------------------------#

batch_size = 25

# attr[img,ind attr]
#print(attr[:,39])



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

features = tf.reshape(tf.stack((young_blond, male_attractive),
                       axis=2),[100,2,2,1])

true_labels = tf.reshape(tf.stack((young_blond, male_attractive),
                          axis=1),[100,4])



data_batch = tf.convert_to_tensor([X_all[idx: idx + batch_size] for idx in
                                   range(0, len(X_all), batch_size)])
features_batch = tf.convert_to_tensor([features[idx: idx + batch_size] for idx in
                                   range(0, len(features), batch_size)])
#print("features _ batch",features_batch.shape)

true_labels_b = tf.convert_to_tensor([true_labels[idx: idx + batch_size] for idx in
                                   range(0, len(true_labels), batch_size)],dtype=tf.float32)





def train(data_batch, attr_batch,true_labels_b, AE, D, AE_optimizer,
          num_epochs=10,
          l=0,AE_MSE=tf.keras.losses.MeanSquaredError(),
          D_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)):

    reconstruction_losses = []
    AE_losses = []
    D_losses = []
    rec_inv_tradeoff = l


    for epoch in range(num_epochs):

        for i in range(data_batch.shape[0]):
            img = data_batch[i]
            attr = features_batch[i]
            true_label = true_labels_b[i]
            inputs = img, attr


            #D.discriminator.trainable = True
            #AE.trainable = False
           
            # Trainable variables are automatically tracked by GradientTape
            with tf.GradientTape(persistent=True) as tape:
                decod,embedding = AE(inputs)
                outD = D(embedding)
                discr_loss = D_loss(true_label,outD)
                #reconstruction_loss = AE_MSE(decod, img)
                #var = 1.0 - true_label
                #model_loss = reconstruction_loss + l * D_loss(outD,var)

            # We use GradientTape to calculate the gradients with respect to discr_loss and trainable_weights of D
            grad = tape.gradient(discr_loss,D.trainable_weights)
            AE_optimizer.apply_gradients(zip(grad,D.trainable_weights))
            #grad_1 = tape.gradient(model_loss, AE.trainable_weights)
            #AE_optimizer.apply_gradients(zip(grad_1,AE.trainable_weights))


        #AE_losses.append(model_loss.numpy())
        D_losses.append(discr_loss.numpy())


        print("epoch: {}/{} epochs loss_D {}  ".format(epoch,
                                                               num_epochs,
                                                        discr_loss.numpy(),
                                                                 ))

    plt.figure(figsize=[12,5])
    plt.subplot(121)
    plt.title('AE_loss ', fontsize=20)
    plt.plot(D_losses)
    plt.grid()

    plt.subplot(122)
    plt.title("D_loss", fontsize=20)
    plt.plot(D_losses)
    plt.grid()
    plt.show()






AE_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5,
                                name='Adam',weight_decay=1e-5)
AE = model.Autoencoder()
D = model.Discreminateur()

#D.compile(AE_optimizer)
#AE.compile(AE_optimizer)


train(data_batch,features_batch,true_labels_b,AE,D,AE_optimizer,
      num_epochs=100,
          l=0.0,AE_MSE=tf.keras.losses.MeanSquaredError(),
          D_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

img = data_batch[0]
attr = features_batch[0]
true_label = true_labels_b[0]
inputs = img, attr
decod,embedding = AE(inputs)

visuelData.Affichge(decod)



