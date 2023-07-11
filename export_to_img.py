import cv2
import numpy as np


test_ds=np.loadtxt('./mnist_test.csv',delimiter=',')
test_img=test_ds[9999,1:].reshape(28,28)
test_lbl=test_ds[9999,0]

cv2.imwrite('./pred_img.png',test_img)
print(test_lbl)