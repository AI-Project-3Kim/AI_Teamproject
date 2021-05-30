#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
import random
import numpy as np
import tensorflow as tf

#데이터 전처리
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    directory='./curated_data',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(224, 224),
    seed=123,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)
test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    directory='./curated_data',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(224, 224),
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)

#lenet 구조
model = Sequential()

model.add(Conv2D(filters = 6, kernel_size = (5,5) , padding="same", input_shape=(224,224,1)))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 16, kernel_size = (5,5) , padding="same"))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())
    
model.add(Dense(120))
model.add(Activation("relu"))

model.add(Dense(84))
model.add(Activation("relu"))

model.add(Dense(3, activation='softmax'))
adam = optimizers.Adam(lr = 0.003)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])



model.summary()




initial_epochs = 10
history = model.fit(
  train_dataset,
  epochs=initial_epochs
)




results=model.evaluate(test_dataset)


import matplotlib.pyplot as plt
#train accuracy print

plt.plot(history.history['accuracy'])
plt.legend(['training'], loc = 'upper left')
plt.show()


print(results)





