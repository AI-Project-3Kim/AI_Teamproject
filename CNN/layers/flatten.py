import numpy as np

class Flatten:

    def __init__(self):
        # 입력받을 데이터의 shape를 받을 변수
        self.prev_shape = ()
        self.dweight = None
        self.dbias = None
        
    def forward(self, prev_arr):
        # Conv, pooling 이 끝난 데이터(n차원)을 입력받아서 1차원으로 변환한다.
        self.prev_shape = prev_arr.shape
        
        return np.ravel(prev_arr).reshape(prev_arr.shape[0], -1)

    def backward(self, after_arr):
        # 변환하기 전의 shape 로 1차원 array를 reshape 한다.
        return after_arr.reshape(self.prev_shape)
    def get_gradient(self):
        if self.dweight is None or self.dbias is None:
            return None
        g_w = self.dweight
        g_b = self.dbias
        return g_w,g_b
