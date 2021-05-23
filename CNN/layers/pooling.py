#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
class MaxPooling:
    def __init__(self,size=(2,2),stride=2):
        #풀링 창 사이즈
        self.size=size
        #스트라이드 VGG 기본 2
        self.stride=stride
        self.mask={}
    def forward(self,input_):
        """
        b_s = batch size
        width_in = input width
        height_in = input height
        ch = channel num
        """
        self.layer_shape=input_.shape
        b_s,width_in,height_in,ch=self.layer_shape
        output_shape=(b_s,(height_in-self.size[0])//self.stride +1,
                      (width_in-self.size[1])//self.stride +1,ch)
        out=np.zeros(output_shape)
        for y in range(output_shape[1]):
            for x in range(output_shape[2]):
                h_s=self.stride*y
                h_e=h_s+self.size[0]
                w_s=self.stride*x
                w_e=w_s+self.size[1]
                
                slice_=input_[:,h_s:h_e,w_s:w_e,:]
                self.make_mask(slice_,(y,x))
                out[:,y,x,:]=np.max(slice_,axis=(1,2))
        return out
    def make_mask(self,array,index):
        tmp=np.zeros_like(array)
        n,height,width,ch=array.shape
        array=array.reshape(n,height*width,ch)
        idx_list=np.argmax(array,axis=1)
        n_idx,c_idx=np.indices((n,ch))
        tmp.reshape(n,height*width,ch)[n_idx,idx_list,c_idx]=1
        self.mask[index]=tmp
    def backward(self,input_):
        out=np.zeros(self.layer_shape)
        n,height_out,width_out=input_.shape
        for y in range(height_out):
            for x in range(width_out):
                w_s=self.stride*x
                w_e=w_s+self.size[1]
                h_s=self.stride*y
                h_e=h_s+self.size[0]
                
                out[:,h_s:h_e,w_s:w_e,:]+=                     input_[:,y:y+1,x:x+1,:]*self.mask[(y,x)]
                

