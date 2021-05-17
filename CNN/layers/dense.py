import numpy as np



class DenseLayer:

    def __init__(self, w, b):
        # flatten layer의 출력 데이터(전)와 4096개의layer(후)
        # flatten layer의 출력 데이터(전)와 클래스별로 분리된 layer(후)
        # weight : (prev_num, after_num)
        self.weight = w
        self.bias = b
        # gradient의 기울기 , gradient의 bias
        self.gradient_weight = None
        self.gradient_bias = None
        # flatten layer의 출력 데이터 
        self.prev_arr = None

    @classmethod
    def initialize(cls, prev_num, after_num):
        # weight의 크기만큼 random으로 초기화시킨다.
        weight_init_std=0.1
        weight = np.random.randn(prev_num, after_num) * weight_init_std
        bias = np.random.randn(1, after_num) * weight_init_std
        return cls(w=weight, b=bias)

    def get_weight(self):
        weight = self.weight
        bias = self.bias
        return weight,bias
  
    def get_gradient(self):
        if self._dw is None or self._db is None:
            return None
        g_w = self.gradient_weight
        g_b = self.gradient_bias
        return g_w,g_b

    def forward(self, prev_arr):
        # flatten layer의 출력 데이터 : prev_arr
        self.prev_arr = np.array(prev_arr, copy=True)
        result = np.dot(prev_arr, self.weight) + self.bias
        return result

    def backward(self, after_arr):
        # n : example 의 개수
        # after_arr : dense layer 을 거친 데이터
        n = self.prev_arr.shape[0]
        self.gradient_weight = np.dot(self.prev_arr.T, after_arr) / n
        # 차원 유지 , 세로 합
        self.gradient_bias = np.sum(after_arr, axis=0, keepdims=True) / n
        # result 는 이 전 층으로 가는 결과
        result =  np.dot(after_arr, self.weight.T)
        return result

    def set_weight(self, w, b):
        self.weight = w
        self.bias = b
