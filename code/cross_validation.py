#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

import matplotlib.pylab as plt


# In[34]:


num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)
print(y_train[0])
y_train = to_categorical(y_train, num_classes)
print(y_train[0])
print(y_train.shape)


# In[35]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


# In[38]:


model = Sequential()
model.add(Dense(10, input_shape=(1,)))

sgd = SGD(lr=0.003)
model.compile(optimizer=sgd, loss=)


# In[ ]:




