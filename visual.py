#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:54:34 2022

 
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
