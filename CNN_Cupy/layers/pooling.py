#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cupy as np
class MaxPooling:
    def __init__(self,size=(2,2),stride=2):
        #풀링 창 사이즈
        self.size=size
        #스트라이드 VGG 기본 2
        self.stride=stride
        self.mask={}
        self.dweight = None
        self.dbias = None
    def forward(self,input_):
        """
        b_s = batch size
        width_in = input width
        height_in = input height
        ch = channel num
        """
        self.layer_shape=input_.shape
        
        b_s,width_in,height_in,ch=self.layer_shape
        #input data shape와 stride를 이용해서 output shape를 계산해줌
        output_shape=(b_s,(height_in-self.size[0])//self.stride +1,
                      (width_in-self.size[1])//self.stride +1,ch)
        out=np.zeros(output_shape)
        
        for y in range(output_shape[1]):
            for x in range(output_shape[2]):
                h_s=self.stride*y
                h_e=h_s+self.size[0]
                w_s=self.stride*x
                w_e=w_s+self.size[1]
                #stride를 이용해 pooling 할만큼 slice해줌
                slice_=input_[:,h_s:h_e,w_s:w_e,:]
                #backward 할 경우를 생각해 mask 저장해줌
                self.make_mask(slice_,(y,x))
                #max 함수를 이용해서 maxpooling 해줌
                out[:,y,x,:]=np.max(slice_,axis=(1,2))
        return out
    
    def make_mask(self,array,index):
        tmp=np.zeros_like(array)
        n,height,width,ch=array.shape
        array=array.reshape(n,height*width,ch)
        idx_list=np.argmax(array,axis=1)
        n_idx,c_idx=np.indices((n,ch))
        tmp.reshape(n,height*width,ch)[n_idx,idx_list,c_idx]=1
        #index를 넣으면 어디가 max였던지 알려줌
        self.mask[index]=tmp
        
    def backward(self,input_):
        out=np.zeros(self.layer_shape)
        n,height_out,width_out,ch=input_.shape
        for y in range(height_out):
            for x in range(width_out):
                w_s=self.stride*x
                w_e=w_s+self.size[1]
                h_s=self.stride*y
                h_e=h_s+self.size[0]
                # maxpool 되었던 곳에 받은 input을 더해줌
                out[:,h_s:h_e,w_s:w_e,:]+=input_[:,y:y+1,x:x+1,:]*self.mask[(y,x)]
        return out        
    def get_gradient(self):
        return None
    def set_weight(self, w, b):
        
        pass
    def get_weight(self):
        
        return None
