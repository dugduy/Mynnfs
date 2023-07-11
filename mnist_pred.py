from pickle import load
from mysimple_autodiff import *
import cv2
from mnist_train import Net
import matplotlib.pyplot as plt

img=cv2.imread('./pred_img.png',cv2.IMREAD_GRAYSCALE)#/255.
mymodel=Net('mymodel')
mymodel.load_weight(load(open('./mnist.pickle','rb')))

print(mymodel(reshape(img,newshape=(1,-1))))
plt.imshow(img)
plt.show()