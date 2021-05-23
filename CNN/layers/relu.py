import numpy as np
import copy

class Relu():
    def __init__(self):
        self.relu_output = None
        self.dweight = None
        self.dbias = None
        
    def forward(self,prev_arr):
        # 입력받을 값 (prev_arr)을 Relu 함수에 적용한다.
        # 0보다 크면 그냥 prev_arr 값 출력, 0보다 작으면 0값 출력한다.
        self.relu_output = np.maximum(0, prev_arr)
        return self.relu_output

    def backward(self, after_arr):
        # forward 에서 Relu 함수 적용한 arr을 역전파시킨다.
        back_arr = copy.deepcopy(after_arr)
        # 0보다 작으면 0, 0보다 크면 그대로
        back_arr[self.relu_output <= 0] = 0
        return back_arr
    def get_gradient(self):
        if self.dweight is None or self.dbias is None:
            return None
        g_w = self.dweight
        g_b = self.dbias
        return g_w,g_b
