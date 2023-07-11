class SGD:
    def __init__(self,paramts,lr=0.001,name='') -> None:
        self.name=name
        self.lr=lr
        self.paramts=paramts
    def optim(self,grad_dict):
        for paramt in self.paramts:
            paramt.assign_sub(grad_dict[paramt]*self.lr)