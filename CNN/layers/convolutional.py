import numpy as np
import copy


class Conv():

    def __init__(self, num_stride, padding, num_filter, filter_size, input_shape):
        self.stride = num_stride
        self.padding = padding  # 'same' or 'valid'
        self.num_filter = num_filter
        self.filter_size = filter_size  # [a,b,c]

        self.weights = np.reshape(
            np.random.normal(0, 0.05, filter_size[0] * filter_size[1] * filter_size[2] * num_filter),
            (filter_size[0], filter_size[1], filter_size[2], num_filter))
        # height, width, channel, num_filter
        self.bias = np.reshape(np.random.normal(0, 0.05, num_filter), (num_filter))
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

        self.input_shape = input_shape  # batch_size, height, width, channel
        self.output_shape = self.get_output_shape(input_shape)  # batch_size, height, width, filter

        self.inputt = np.zeros_like(input_shape)
        
        self.padding_shape = self.get_pad_shape()
        # self.input_col = None

    def get_output_shape(self, input_shape):  # batch_size, height, width, channel
        if self.padding == 'same':
            return [input_shape[0], input_shape[1], input_shape[2], self.num_filter]
            # batch number, height number, width number, filter number
        elif self.padding == 'valid':
            return [input_shape[0], (input_shape[1] - weights.shape[0]) // self.stride + 1,
                    (input_shape[2] - weights.shape[1]) // self.stride + 1, self.num_filter]

    def get_weight(self):
        return self.weights, self.bias

    def get_gradient(self):
        return self.dweights, self.dbias

    def set_weight(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def set_gradient(self, dw, db):
        self.dweights  = dw
        self.dbias = db

    def get_pad_shape(self):
        if self.padding == 'same':
            return ((self.weights.shape[0] - 1) // 2, (self.weights.shape[1] - 1) // 2)
        elif self.padding == 'valid':
            return (0, 0)

    def forward(self, inputt):
        self.inputt = copy.deepcopy(inputt)
        output = np.zeros(self.output_shape)
        # print(self.output_shape)
        input_padded = np.pad(inputt, pad_width=(
        (0, 0), (self.padding_shape[0], self.padding_shape[0]), (self.padding_shape[1], self.padding_shape[1]), (0, 0)))
        print(input_padded.shape)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                height_start = i * self.stride
                height_end = height_start + self.weights.shape[0]
                width_start = j * self.stride
                width_end = width_start + self.weights.shape[1]

                output[:, i, j, :] = np.sum(
                    input_padded[:, height_start:height_end, width_start:width_end, :, np.newaxis] *
                    self.weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        return output + self.bias

    def backward(self, loss):  # batch, height, width, num_filter

        input_padded = np.pad(inputt, pad_width=(
        (0, 0), (self.padding_shape[0], padding_shape[0]), (padding_shape[1], padding_shape[1]), (0, 0)))
        output = np.zeros_like(input_padded)

        for i in range(loss.shape[1]):
            for j in range(loss.shape[2]):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.weights[np.newaxis, :, :, :, :] *
                    loss[:, i:i + 1, j:j + 1, np.newaxis, :],
                    axis=4
                )
                self.dweights += np.sum(
                    input_padded[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    loss[:, i:i + 1, j:j + 1, np.newaxis, :],
                    axis=0
                )

        self.dweights /= n
        return output[:, pad[0]:pad[0] + self.inputt.shape[1], pad[1]:pad[1] + self.inputt.shape[2], :]
