import cupy as np

class Adam():
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):
        """
        m = moment mean dict
        v = moment variance dict
        beta_1, beta_2 = parameters of Adam
        eps = epsilon
        """
        self.m = {}
        self.v = {}
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        
    def update(self,layers):
        #없으면 init 해줌
        if not len(self.m) or not len(self.v):
            for i,layer in enumerate(layers):
                g=layer.get_gradient()
                if not g:
                    continue
                dw,db=g
                dw_idx="dw"+str(i)
                db_idx="db"+str(i)
                
                self.m[dw_idx]=np.zeros_like(dw)
                self.m[db_idx]=np.zeros_like(db)
                self.v[dw_idx]=np.zeros_like(dw)
                self.v[db_idx]=np.zeros_like(db)
        
        #받아온다
        for i,layer in enumerate(layers):
            weights,gradients=layer.get_weight(),layer.get_gradient()
            (w,b)=weights
            (dw,db)=gradients
            
            #key 값 
            dw_idx="dw"+str(i)
            db_idx="db"+str(i)
            
            self.m[dw_idx]=self.beta_1*self.m[dw_idx]+(1-self.beta_1)*dw
            self.m[db_idx]=self.beta_1*self.m[db_idx]+(1-self.beta_1)*db
            
            self.v[dw_idx]=self.beta_2*self.v[dw_idx]+(1-self.beta_2)*np.power(dw,2)
            self.v[db_idx]=self.beta_2*self.v[db_idx]+(1-self.beta_2)*np.power(db,2)
            # 나중에 가중치 편향이 일어나면 step 고려해줘야함
            dw=self.m[dw_idx]/(np.sqrt(self.v[dw_idx])+self.eps)
            db=self.m[db_idx]/(np.sqrt(self.v[db_idx])+self.eps)
            
            weight=layer.weights-self.lr*dw
            bias=layer.bias-self.lr*db
            
            layer.set_weight(weight,bias)

