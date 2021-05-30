#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import random
import numpy as np
import tensorflow as tf

#데이터 
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


#input
inputs = tf.keras.layers.Input(shape=(1,))
#predictions
predictions = tf.keras.layers.Dense(1)(inputs)
#VGG16
model = tf.keras.applications.VGG16(weights=None,input_shape=(224,224,1),classes=3)
#Adam Optimizer
adam = optimizers.Adam(learning_rate = 0.003)
#loss function -> categorical_crossentropy
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




