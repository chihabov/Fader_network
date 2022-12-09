#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:15:05 2022

@author: cherif
"""
import numpy as np
import pandas as pd
csv_path=csv_path = '../images/list_attr_celeba.csv'

def create_attributes(csv_path,nbr_data):
    attr=open(csv_path)
    attr=attr.readlines()
    attributes=[]
    for j in range(1,nbr_data):
           attrib=attr[j][10:-1].split(",")
            
           attrib = list(attrib)
            
           attributes.append(attrib)
    return attributes
def switch_att(y):
  y_=[]
  for i in y:
    i_=[]
    for j in range(1,len(i)):
      try:
        if i[j]=='1':
           i_.append(1)
        if  i[j]=='-1':
          i_.append(0)
      except:
        pass
    y_.append(i_)
  return np.array(y_)
def attr_ind(x, data):

    df = pd.read_csv(csv_path)
    file = df.to_numpy()
    
    ind_attr = x
    A = []
    for i in range(len(data)):
        A.append(file[i][ind_attr])
    return A
