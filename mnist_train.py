from lossfn import MSE
from mynn import Module
from layers import Dense
from optim import SGD
from activations import Sigmoid,ReLU
from mysimple_autodiff import *

np.warnings.filterwarnings('ignore')
class Net(Module):
    def __init__(self, name='') -> None:
        super().__init__(name)
        self.d1=Dense(784,128,'d1')
        self.d2=Dense(128,10,'d2')
        self.relu=ReLU('relu')
        self.sigmoid=Sigmoid('sigmoid')
    def __call__(self, x, training=False):
        x=self.relu(self.d1(x))
        return self.sigmoid(self.d2(x))

# net=Net('mymodel')
# loss_fn=MSE('mse')
# optimizer=SGD(net.trainable_paramts,name='sgd')
# print('Trainable paramts:',net.trainable_paramts)

# one_hot_encoder=np.eye(10)
# train_ds=np.loadtxt('./mnist_train.csv',delimiter=',',dtype='int')
# imgs_train=Variable(train_ds[:,1:])/255.0
# lbls_train=Variable(one_hot_encoder[train_ds[:,0]])
# test_ds=np.loadtxt('./mnist_test.csv',delimiter=',',dtype=int)
# imgs_test=Variable(test_ds[:,1:])/255.0
# lbls_test=Variable(one_hot_encoder[test_ds[:,0]])

# def test(imgs,lbls):
#     pred=net(imgs,training=False)
#     loss=loss_fn(pred,lbls)/imgs.shape[0]
#     accuracy=np.mean(np.argmax(pred.value,1)==np.argmax((lbls.value),1))
#     print(f'loss: {loss}    accuracy: {accuracy}')
# print('------------------------------------------------------------------\nTrainset test')
# test(imgs_train,lbls_train)
# print('Testset test')
# test(imgs_test,lbls_test)

# batched_train_imgs=reshape(imgs_train,newshape=(1875,32,784))
# batched_train_lbls=reshape(lbls_train,newshape=(1875,32,10))

# for epoch in range(5):
#     print('Epoch',epoch+1)
#     for i in range(1875):
#         print(f'step {i+1}/1875',end='\r')
#         x=batched_train_imgs[i]
#         y=batched_train_lbls[i]
#         pred=net(x)
#         loss=loss_fn(pred,y)
#         optimizer.optim(gradients(loss))
#     print('-------------------------\nTrainset test')
#     test(imgs_train,lbls_train)
#     print('Testset test')
#     test(imgs_test,lbls_test)
# print('New trainable paramts:',net.trainable_paramts)
# net.save_weight('mnist.pickle')