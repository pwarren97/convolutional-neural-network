#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense


# In[ ]:


# Conv2D(filters, kernel_size, strides=(1,1), padding='valid', data_format=None, dilation_rate=(1,1), activation=None, use_bias=True, kernel_intitializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_contraint=None, bias_contraint=None),
# MaxPooling2D(poolsize=(2, 2), strides=2, padding='valid', data_format=None),
# Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Dense(number, 'relu'))


# In[ ]:


# Compile follows setting up the neural network
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

