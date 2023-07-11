from mysimple_autodiff import *
class BaseLayer:
    def __init__(self,trainable_params=[],name='') -> None:
        self.trainable_paramts=trainable_params
        self.name=name
    def __call__(self,x,training=False):
        return
    
class Dense(BaseLayer):
    def __init__(self,n_inputs,n_neurons, name='') -> None:
        self.w=Variable(np.random.randn(n_inputs,n_neurons)*0.1,name+'.w')
        self.b=Variable(np.zeros(n_neurons),name+'.b')
        super().__init__([self.w,self.b],name)
    def __call__(self, x):
        self.output=x@self.w+self.b
        return self.output