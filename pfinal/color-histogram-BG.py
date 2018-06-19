import numpy as np
import scipy.spatial as sp
import cv2
import glob
import matplotlib.pyplot as plt
import time



# return arrays of images ORI, GT
def loadImages(I, color=cv2.COLOR_BGR2RGB):
  Iout=[]
  for j in range (0,len(I)):
    print (I[j])
    Iout.append(cv2.cvtColor(cv2.imread(I[j]), color))
  return Iout


def getBitwiseMatrix(Y):
  Y = np.asarray(Y)
  threshold = 1
  mask = np.where((Y[:,:,:, 0] >= threshold) &
                  (Y[:,:,:, 1] >= threshold) &
                  (Y[:,:,:, 2] >= threshold))                
  Y[:,:,:,:] = 0
  Y[mask] = 1
  
  return Y

def preProcessImages(X, Y):
  print ('.')
  X = np.asarray(X)
  Y = np.asarray(Y)
  threshold = 0
  print (Y.shape)
  mask = np.where(Y[:, :, :, 1] >= 254)
  Y[:, :, :, :] = 0
  print('.')
  Y[mask] = 255
  skinMask = Y
  skinMask[mask] = 1
  skin = cv2.bitwise_and(X, X, mask=skinMask)
  print('.')
  notSkinMask = Y
  notSkinMask[:, :, :, :] = 1
  notSkinMask[mask] = 0
  notSkin = cv2.bitwise_and(X, X, mask=notSkinMask)
  print('.')
  return (skinMask, notSkinMask, skin, notSkin)


def extractHist(sets, categories, color, name="RGB"):
  
  for j, images in enumerate(sets):
    u = images[:,:,:,1:2]
    
    print(u[1], u[0])
    # print(images[:, :, :, 2].shape)
    # images[:,:,:,0]=uv
    # images = images[:, :, :, 0]
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist
    # print (images.shape)
  #   hist = cv2.calcHist(images, [1:2], None, [256*256], [0, 256*256])
  #   plt.plot(hist, label=categories[j])
  #   plt.xlim([0, 256])
  # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
  #             ncol=2, mode="expand", borderaxespad=0.)
  # plt.title(name+"  "+col)
  # plt.minorticks_on()
  # plt.grid(which='major', linestyle='-', linewidth='0.75', color='black')
  # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
  # print('./results/{}-{}-1.png'.format(name, col))
  # plt.savefig('./results/{}-{}-1.png'.format(name,col))
  # plt.gcf().clear()

def generateHistograms():
  colors = ["YCrCb", cv2.COLOR_BGR2YCrCb,  ("Y", "Cr", "Cb")]


  color = colors[1]
  
  Xtrain = loadImages(I=trainImages, color=color)
  Ytrain = loadImages(I=trainGTs, color=color)
  
  # pre-process images
  skinMask, notSkinMask, skin, notSkin = preProcessImages(Xtrain[5:], Ytrain[5:])
  print ('extracting')
  # generate histogram
  extractHist([skin, notSkin], categories=('skin', 'not skin'), color=colors[2], name=colors[0])



# import datasets
trainImages = glob.glob('./dataset/*/*.jpg')[0:30]
trainGTs = glob.glob('./BG/*/*.jpg')[0:30]


generateHistograms()
