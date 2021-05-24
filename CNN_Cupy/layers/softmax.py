import cupy as np

class Softmax():
    
    def __init__(self):
        self.softmax_output= None
        self.dweight = None
        self.dbias = None
        self.weights = None
        self.bias= None
        
    def forward(self, prev_arr):
        # prev_arr을 softmax 함수를 처리한다.
        # 매우 큰 수가 되면 계산에 문제가 될 수 있기 때문에 최댓값을 빼준다. (결국 같은 값)
        # keepdims : 차원 유지 시켜준다.
        # axis =1 : 열 별로 계산 - 행 별로 최댓값을 구해서 빼준다.
        exp_prev_arr = np.exp(prev_arr - prev_arr.max(axis=1, keepdims=True))
        # exp 한 값의 전체 합
        exp_prev_arr_sum =np.sum(exp_prev_arr, axis=1, keepdims=True)
        #softmax 한 결과
        self.softmax_output = exp_prev_arr / exp_prev_arr_sum
        return self.softmax_output

    def backward(self, after_arr):
        # softmax 값 - target 값 한 결과가 softmax loss의 역전파 
        # (Sequential.py에 시행)
        return after_arr
    
    def get_gradient(self):
        if self.dweight is None or self.dbias is None:
            return None
        g_w = self.dweight
        g_b = self.dbias
        return g_w,g_b
    
    def get_weight(self):
        weight = self.weights
        bias = self.bias
        return weight,bias

    def set_weight(self, w, b):
        self.weights = w
        self.bias = b
