import cupy as np
import copy
from typing import Tuple

class ConvLayer2D(Layer):

    def __init__(
        num_stride: int = 1,
        padding: str = 'valid',
        self, w: np.array,
        b: np.array,
    ):
        self.weights, self.bias= w, b
        self.padding = padding
        self.stride = stride
        self.dweights, self.dbias = None, None
        self._a_prev = None

    @classmethod
    def initialize(
        cls, filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        stride: int = 1
    ) -> ConvLayer2D:
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @property
    def get_weight(self) -> Optional[Tuple[np.array, np.array]]:
        return self.weights, self.bias

    @property
    def get_gradient(self) -> Optional[Tuple[np.array, np.array]]:
        if self.dweights is None or self.dbias is None:
            return None
        return self.dweights, self.dbias

    def forward(self, a_prev: np.array) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in, w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self.weights.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f

                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self.weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        return output + self.bias

    def backward(self, da_curr: np.array) -> np.array:
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self._a_prev.shape
        h_f, w_f, _, _ = self.weights.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=self._a_prev, pad=pad)
        output = np.zeros_like(a_prev_pad)

        self.dbias = da_curr.sum(axis=(0, 1, 2)) / n
        self.dweights = np.zeros_like(self.weights)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.weights[np.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self.dweights += np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self.dweights /= n
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]

    def set_weight(self, w: np.array, b: np.array) -> None:
        self.weights = w
        self.bias = b

    def calculate_output_dims(
        self, input_dims: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self.weights.shape
        if self.padding == 'same':
            return n, h_in, w_in, n_f
        elif self.padding == 'valid':
            h_out = (h_in - h_f) // self.stride + 1
            w_out = (w_in - w_f) // self.stride + 1
            return n, h_out, w_out, n_f
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self.padding}"
            )

    def calculate_pad_dims(self) -> Tuple[int, int]:
        if self.padding == 'same':
            h_f, w_f, _, _ = self.weights.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self.padding == 'valid':
            return 0, 0
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self.padding}"
            )

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]) -> np.array:
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )