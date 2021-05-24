import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from CNN_Cupy.layers.convolutional import Conv
from CNN_Cupy.layers.pooling import MaxPooling
from CNN_Cupy.layers.flatten import Flatten
from CNN_Cupy.layers.dense import DenseLayer
from CNN_Cupy.layers.softmax import Softmax
from CNN_Cupy.layers.relu import Relu
from CNN_Cupy.layers.dropout import Dropout
from CNN_Cupy.optimizer.adam import Adam
from CNN_Cupy.sequential import SequentialModel

### data

x = np.load('./Dataset/x.npy')
y_data = np.load('./Dataset/y.npy')

y=np.array([[0,0,0]])
for i in y_data:
    if i == 0:
        label=np.array([1,0,0])
    elif i == 1:
        label=np.array([0,1,0]) 
    else:
        label=np.array([0,0,1])  
    label=label.reshape(1,3)
    y=np.concatenate([y,label],axis=0)
# 이게 one-hot vector 값 ( 처음에 초기화 한거 지움 )
y=np.delete(y,0,axis=0)

test_idx = random.sample(range(len(x)),int(0.3*len(x)))
x_test = x[test_idx,:,:]
x_train = np.delete(x, test_idx, axis=0)
y_test = y[test_idx,:]
y_train = np.delete(y, test_idx, axis=0)


x_train = x_train.reshape(len(x_train), 224,224,1)
x_test = x_test.reshape(len(x_test), 224,224,1)


import cupy as np
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

bs=64
layers = [
    Conv(num_stride=1, padding = "same", num_filter=64, filter_size=(3, 3, 1), input_shape = (bs, 224,224,1)),
    Relu(),
    MaxPooling(size=(2, 2), stride=2),
    Dropout(dropout_rate=0.75),
    
    Conv(num_stride=1, padding = "same", num_filter=128, filter_size=(3, 3, 64), input_shape = (bs, 112,112,64)),
    Relu(),
    MaxPooling(size=(2, 2), stride=2),
    Dropout(dropout_rate=0.75),
    Flatten(),
    DenseLayer.initialize(prev_num=56*56*128, after_num=3),
    Softmax()
]
optimizer = Adam(lr=0.003)
model = SequentialModel(
    layers=layers,
    optimizer=optimizer
)

epochs = 20
model.train(x_train = x_train, y_train = y_train, epochs = epochs, batch_size = bs)
model.test(x_test = x_test, y_test = y_test, batch_size = bs)


plt.plot(range(epochs), model.train_accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()
plt.savefig('./accuracy.png')

plt.plot(range(epochs), model.train_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
plt.savefig('./loss.png')