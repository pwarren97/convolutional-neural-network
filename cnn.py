#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense


# In[13]:


# Data setup
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape into 4D tensor with tensorflow reshape function
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# In[ ]:


model = Sequential()


# In[4]:


# Add layers
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))


# In[ ]:


# Compile follows setting up the neural network
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

