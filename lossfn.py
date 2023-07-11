from mysimple_autodiff import *

class BaseLoss:
    def __init__(self,name='') -> None:
        self.name=name
    def __call__(self,y_pred,y_true):
        return
    
class MSE(BaseLoss):
    def __call__(self,y_pred,y_true):
        return reduce_sum((y_pred-y_true)**2)

class CE(BaseLoss):
    def __call__(self, y_pred, y_true):
        eps=1e-12
        y_pred=clip(y_pred,eps,1-eps)
        return -reduce_sum(y_true*log(y_pred),axis=1)