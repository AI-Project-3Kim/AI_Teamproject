import numpy as np

# 네트워크가 과적합되는 경우를 방지하기 위해서 만들어진 레이어
# 학습 과정에서 무작위로 뉴런의 집합을 제거하는 것이 과적합을 막는다는 아이디어
# 학습 과정에서는 일부 뉴런을 제거하여 일부 영향력을 줄여서 학습하고, 테스트에서는 모든 값을 포함하여 계산하는 방법

class Dropout:

    def __init__(self, dropout_rate):
        # 몇 개만큼 노드를 쓸지 
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.dweight = None
        self.dbias = None
        self.weight = None
        self.bias = None
        
    def forward(self, prev_arr, trainflg=True):
        # train 할 때 dropout을 적용해야한다.
        if trainflg:
            # dropout_rate 만큼의 arr을 선택
            self.dropout_mask = np.random.rand(*prev_arr.shape) < self.dropout_rate
            return prev_arr * self.dropout_mask / self.dropout_rate            
        else:
            # test 할 때는 dropout 하지 않음
            return prev_arr

    def backward(self, after_arr):
        return after_arr * self.dropout_mask / self.dropout_rate
    
    def get_gradient(self):
        if self.dweight is None or self.dbias is None:
            return None
        g_w = self.dweight
        g_b = self.dbias
        return g_w,g_b
    def get_weight(self):
        weight = self.weight
        bias = self.bias
        return weight,bias

    def set_weight(self, w, b):
        self.weight = w
        self.bias = b
