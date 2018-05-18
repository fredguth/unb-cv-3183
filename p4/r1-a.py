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
  X = np.asarray(X)
  Y = np.asarray(Y)
  threshold = 27

  mask = np.where((Y[:, :, :, 0] >= threshold) &
                  (Y[:, :, :, 1] >= threshold) &
                  (Y[:, :, :, 2] >= threshold))
  Y[:, :, :, :] = 0
  Y[mask] = 255
  skinMask = Y
  skinMask[mask] = 1
  skin = cv2.bitwise_and(X, X, mask=skinMask)
  notSkinMask = Y
  notSkinMask[:, :, :, :] = 1
  notSkinMask[mask] = 0
  notSkin = cv2.bitwise_and(X, X, mask=notSkinMask)
  return (skinMask, notSkinMask, skin, notSkin)


def extractHist(sets, categories, color, name="RGB"):
  for i, col in enumerate(color):
    for j, images in enumerate(sets):
      # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist
      hist = cv2.calcHist(images, [i], None, [256], [28, 256])
      plt.plot(hist, label=categories[j])
      plt.xlim([0, 256])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.title(name+"  "+col)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.75', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.savefig('./results/{}-{}-1.png'.format(name,col))
    plt.gcf().clear()

def generateHistograms():
  colors = [
      ["YCrCb", cv2.COLOR_BGR2YCrCb,  ("Y", "Cr", "Cb")],
      ["HSV",   cv2.COLOR_BGR2HSV,    ("H", "S", "V")],
      ["LAB",   cv2.COLOR_BGR2LAB,    ("L", "A", "B")],
      ["RGB",   cv2.COLOR_BGR2RGB,    ("R", "G", "B")]
  ]

  for i in range(0, len(colors)):
    color = colors[i][1]
    # load images
    Xtrain = loadImages(I=trainImages, color=color)
    Ytrain = loadImages(I=trainGTs, color=color)

    # pre-process images
    skinMask, notSkinMask, skin, notSkin = preProcessImages(Xtrain, Ytrain)

    # generate histogram
    extractHist([skin, notSkin], categories=('skin', 'not skin'), color=colors[i][2], name=colors[i][0])



# import datasets
trainImages = glob.glob('./dataset/train/ORI/*.jpg')
trainGTs = glob.glob('./dataset/train/GT/*.jpg')
testImages = glob.glob('./dataset/test/ORI/*.jpg')
testGTs = glob.glob('./dataset/test/GT/*.jpg')


generateHistograms()