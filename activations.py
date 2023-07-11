from mysimple_autodiff import *
from mynn import BaseLayer

class ReLU(BaseLayer):
    def __init__(self, name='') -> None:
        self.name=name
    def __call__(self,x,training=False):
        return maximum(x,0)

class Sigmoid(BaseLayer):
    def __init__(self, name='') -> None:
        self.name=name
    def __call__(self,x,training=False):
        return 1/(1+np.e**-x)

class SoftMax(BaseLayer):
    def __init__(self, name='') -> None:
        self.name=name
    def __call__(self, x, training=False):
        norm_x=np.e**x
        return norm_x/reduce_sum(norm_x,axis=1)