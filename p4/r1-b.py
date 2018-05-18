import numpy as np
import scipy.spatial as sp
import cv2
import glob
import matplotlib.pyplot as plt
import time



# return arrays of images ORI, GT
def loadImages(I):
  Iout=[]
  for j in range (0,len(I)):
    Iout.append(cv2.imread(I[j]))
  return Iout



# Prediction and Truth are 0 or 1
def calcAccuracy():
  T=loadImages(I = testGTs)
  T = np.asarray(T)
  T = T[:,:,:,0]
  misses = len(np.where(T > 0)[0])
  P = T
  P[:,:,:]=0
  total = T.size #size of P, T or match
  loss = misses/total
  acc = 1 - loss
  print("Loss {:.2f}%".format(loss*100))
  print ("Accuracy: {:.2f}%".format(acc*100))
  I = P
  U = misses
  I = 0
  U = np.sum(U)
  
  jacc = I/U
  print("Jaccard: {:.2f}%".format(jacc*100))
  return loss, acc, jacc


# import datasets

testImages = glob.glob('./dataset/test/ORI/*.jpg')
testGTs = glob.glob('./dataset/test/GT/*.jpg')

print ('\nNull Algorithm')
print ('='*20)



calcAccuracy()
# print('{:.3f} s'.format(duration))

#fazer com um canal (azul>x como sendo fundo)
#fazer chute de que
