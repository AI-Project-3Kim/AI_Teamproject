import numpy as np
import time
import tqdm

class SequentialModel:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer
        self.train_accuracy = []
        self.test_accuracy = None
        self.train_loss = []
        self.test_loss = None

    def train(self, x_train, y_train, epochs, batch_size):
        for e in range(epochs):
            start = time.time()
            y_hat = np.zeros_like(y_train)
            for idx in range(int(x_train.shape[0]/batch_size)):
                x_train_batch = x_train.take(indices = range(idx, min(idx+batch_size, x_train.shape[0])), axis=0)
                y_train_batch = y_train.take(indices=range(idx, min(idx + batch_size, y_train.shape[0])), axis=0)

                y_hat_batch = self.forward(x_train_batch, y_train_batch, batch_size)
                loss_batch = y_hat_batch - y_train_batch
                self.backward(loss_batch)
                self.update()

                y_hat[idx*batch_size : y_hat_batch.shape[0], :] = y_hat_batch

            self.train_accuracy.append( (np.argmax(y_hat, axis=1) == y_train).all(axis=1).mean() )
            self.train_loss.append( -np.sum(y_train * np.log(np.clip(y_hat, 1e-20, 1.))) / y_hat.shape[0])

            h, r = divmod(start - time.time(), 3600)
            m, s = divmod(r, 60)
            time_per_epoch = "{:0>2}:{:0>2}:{:05.2f}".format(int(h), int(m), s)
            print("iter: {:05} | test loss: {:.5f} | test accuracy: {:.5f} | time: {}"
                  .format(e + 1, self.train_loss, self.train_accuracy, time_per_epoch))

    def test(self, x_test, y_test, batch_size):
        start = time.time()
        y_hat = np.zeros_like(y_test)
        for idx in range(int(x_test[0]/batch_size)):
            x_test_batch = x_test.take(indices = range(idx, min(idx+batch_size, x_test.shape[0])), axis=0)
            y_test_batch = y_test.take(indices=range(idx, min(idx + batch_size, y_test.shape[0])), axis=0)

            y_hat_batch = self.forward(x_test_batch, y_test_batch, batch_size)
            loss_batch = y_hat_batch - y_test_batch

            y_hat[idx*batch_size : y_hat_batch.shape[0], :] = y_hat_batch

        self.test_accuracy = (np.argmax(y_hat, axis=1) == y_test).all(axis=1).mean()
        self.test_loss = -np.sum(y_test * np.log(np.clip(y_hat, 1e-20, 1.))) / y_hat.shape[0]

        h, r = divmod(start - time.time(), 3600)
        m, s = divmod(r, 60)
        time_per_epoch = "{:0>2}:{:0>2}:{:05.2f}".format(int(h), int(m), s)
        print("test loss: {:.5f} | test accuracy: {:.5f} | time: {}".format(self.test_loss, self.test_accuracy, time_per_epoch))

    def forward(self, inputt):
        x = inputt
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, outputt):
        loss = outputt
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def update(self):
        self.optimizer.update(layers=self.layers)
