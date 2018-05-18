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


def showImages(X, Y):
  
  for j in range(0, len(X)):  
    x = X[j]
    y = Y[j]
    
    index = j*2+1
    plt.subplot(2, len(X), index)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x)
    plt.subplot(2, len(X), index+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y)
  plt.subplots_adjust(wspace=0, hspace=0, left=0.1,
                      right=0.9, top=0.9, bottom=0.5)
  plt.show()


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
      hist = cv2.calcHist(images, [i], None, [256], [32, 256])
      plt.plot(hist, label=categories[j])
      plt.xlim([0, 256])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.title(name+"  "+col)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.75', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.savefig('./results/{}-{}.png'.format(name,col))
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


def getBGRMask(images):
  
  X = loadImages(I=images) 
  X = np.asarray(X)
  B = X[:,:,:,0]
  mask = np.zeros((X.shape), dtype=np.uint8)
  mask[B < 60] = 1
  
  image = cv2.bitwise_and(X, X, mask = mask)

  return X, mask, image

def getYCrCbMask(images):
  # Based on analisys of histograms, best color space is YCrCb and lower and upper bounds
  lower = (0, 115, 0)
  upper = (255, 255, 115)
  X = loadImages(I=images, color=cv2.COLOR_BGR2YCrCb)
  YCrCbMask = X
  YCrCbImage = X
  X = np.asarray(X)
  for i in range(0, len(X)):
    image = X[i]
    mask = cv2.inRange(image, lower, upper)
    YCrCbMask[i] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    X[i] = cv2.cvtColor(X[i], cv2.COLOR_YCrCb2RGB)
    YCrCbImage[i] = cv2.bitwise_and(X[i], X[i], mask=mask)
  return X, YCrCbMask, YCrCbImage


def getLABMask(images):
  # Based on analisys of histograms, best color space is YCrCb and lower and upper bounds
  lower = (130, 110, 130)
  upper = (255, 255, 255)
  X = loadImages(I=images, color=cv2.COLOR_BGR2LAB)
  LABMask = X
  LABImage = X
  X = np.asarray(X)
  for i in range(0, len(X)):
    image = X[i]
    mask = cv2.inRange(image, lower, upper)
    LABMask[i] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    X[i] = cv2.cvtColor(X[i], cv2.COLOR_LAB2RGB)
    LABImage[i] = cv2.bitwise_and(X[i], X[i], mask=mask)
  return X, LABMask, LABImage

# Prediction and Truth are 0 or 1
def calcAccuracy(Prediction, Truth):
  P = getBitwiseMatrix(Prediction)
  T = getBitwiseMatrix(Truth)
  miss = cv2.bitwise_xor(P,T)
  misses= np.sum(miss)
  total = T.size #size of P, T or match
  loss = misses/total
  acc = 1 - loss
  print("Loss {:.2f}%".format(loss*100))
  print ("Accuracy: {:.2f}%".format(acc*100))
  I = cv2.bitwise_and(P, T)
  U = cv2.bitwise_or (P,T)
  I = np.sum(I)
  U = np.sum(U)
  jacc = I/U
  print("Jaccard: {:.2f}%".format(jacc*100))
  return loss, acc, jacc


# import datasets
trainImages = glob.glob('./dataset/train/ORI/*.jpg')
trainGTs = glob.glob('./dataset/train/GT/*.jpg')
testImages = glob.glob('./dataset/test/ORI/*.jpg')
testGTs = glob.glob('./dataset/test/GT/*.jpg')

print('\nNaive BGR')
print('='*20)

start = time.time()
X, mask, skin = getBGRMask(testImages)
Y = loadImages(I=testGTs)

mask = np.asarray(mask)
mask[mask > 0] = 1
end = time.time()
duration = end - start
calcAccuracy(Y, mask)
np.save('results/lim-BGRMask.npy', mask)
print('{:.3f} s'.format(duration))
print ('\nNaive YCrCb Filter')
print ('='*20)

start = time.time()
X, mask, skin = getYCrCbMask(testImages)
Y = loadImages(I=testGTs)

mask = np.asarray(mask)
mask[mask>0]=1
end = time.time()
duration = end - start
calcAccuracy(Y, mask)
np.save('results/lim-YCbCrMask.npy', mask)
print ('{:.3f} s'.format(duration))


# print('\nNaive LAB Filter')
# print('='*20)

# start = time.time()
# X, mask, skin = getLABMask(testImages)
# Y = loadImages(I=testGTs)

# mask = np.asarray(mask)
# mask[mask > 0] = 1
# end = time.time()
# duration = end - start
# calcAccuracy(Y, mask)
# print('{:.3f} s'.format(duration))


