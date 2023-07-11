from mysimple_autodiff import *
from layers import BaseLayer
from pickle import dump

class Module:
    def __setattr__(self,name,tg):
        if isinstance(tg,BaseLayer) and hasattr(tg,'trainable_paramts'):
            for paramt in tg.trainable_paramts:
                self.trainable_paramts.append(paramt)
        super().__setattr__(name,tg)
    def __init__(self,name='') -> None:
        self.name=name
        self.trainable_paramts=[]
    def __call__(self,x,training=False):
        return
    def save_weight(self,fn):
        normed_weight=[i.value.tolist() for i in self.trainable_paramts]
        dump(normed_weight,open(fn,'wb'))
    def load_weight(self,normed_weight):
        for value,paramt in zip(normed_weight,self.trainable_paramts):
            paramt.assign(np.array(value))