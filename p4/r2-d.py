import numpy as np
import scipy.spatial as sp
import cv2
import glob
import matplotlib.pyplot as plt
import time
from sklearn import svm
import gc


# return arrays of images ORI, GT
def loadImages(I, color=cv2.COLOR_BGR2RGB):
  Iout = []
  imageFolder = './sfa/ORI/'
  truthFolder = './sfa/GT/'
  training = range(1, 783)
  testing = range(951, 1119)
  if (I == "trainImages"):
    for j in training:
      filename = "{}img ({}).jpg".format(imageFolder, j)
      Iout.append(cv2.cvtColor(cv2.imread(filename), color))
  elif (I == "trainGTs"):
    for j in training:
      filename = "{}img ({}).jpg".format(truthFolder, j)
      Iout.append(cv2.cvtColor(cv2.imread(filename), color))
  elif (I == "testImages"):
    for j in testing:
      filename = "{}img ({}).jpg".format(imageFolder, j)
      Iout.append(cv2.cvtColor(cv2.imread(filename), color))
  elif (I == "testGTs"):
    for j in testing:
      filename = "{}img ({}).jpg".format(truthFolder, j)
      Iout.append(cv2.cvtColor(cv2.imread(filename), color))
  else:
    print("Not valid dataset")

  return Iout

# I is an array of images
# return
#   s = sample size
#   h = image height
#   w = image width
#   c = color
#   A = striped image 2d array [pixelindex, color]
def image2Data(I):
  I = np.asarray(I)
  s, h, w, c = I.shape
  I = np.reshape(I, (s*h*w, c))
  return s,h,w,c,I


def show(X):
  for j in range(0, len(X)):
    plt.subplot(len(X),1, j+1)
    plt.imshow(X[j])
  plt.show()

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

def showData(X, Y, s, h, w):
  X = np.reshape(X, (s,h,w,-1))
  Y = np.reshape(Y, (s, h, w, -1))
  showImages(X,Y)
 

def getBitwiseMatrix(Y):
  Y = np.asarray(Y)
  threshold = 1
  mask = np.where((Y[:,:,:, 0] >= threshold) &
                  (Y[:,:,:, 1] >= threshold) &
                  (Y[:,:,:, 2] >= threshold))                
  Y[:,:,:,:] = 0
  Y[mask] = 1
  
  return Y


def getSkin(X, Y):
  X = np.asarray(X)
  Y = np.asarray(Y)
  threshold = 27

  mask = np.where((Y[:, :, :, 0] >= threshold) &
                  (Y[:, :, :, 1] >= threshold) &
                  (Y[:, :, :, 2] >= threshold))
  skinMask = np.zeros((Y.shape), dtype=np.uint8)
  skinMask[mask] = 1
  skin = cv2.bitwise_and(X, X, mask=skinMask)

  return skinMask, skin


def getNotSkin(X, Y):
  X = np.asarray(X)
  Y = np.asarray(Y)
  threshold = 27

  mask = np.where((Y[:, :, :, 0] < threshold) &
                  (Y[:, :, :, 1] < threshold) &
                  (Y[:, :, :, 2] < threshold))
  notSkinMask = np.zeros((Y.shape), dtype=np.uint8)
  notSkinMask[mask] = 1
  notSkin = cv2.bitwise_and(X, X, mask=notSkinMask)

  return notSkinMask, notSkin

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
    Xtrain = loadImages(I="trainImages", color=color)
    Ytrain = loadImages(I="trainGTs", color=color)

    # pre-process images
    skinMask, notSkinMask, skin, notSkin = preProcessImages(Xtrain, Ytrain)

    # generate histogram
    extractHist([skin, notSkin], categories=('skin', 'not skin'), color=colors[i][2], name=colors[i][0])

def getYCrCbMask(images):
  # Based on analisys of histograms, best color space is YCrCb and lower and upper bounds
  lower = (110, 115, 0)
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
def train():
  X = loadImages(I="trainImages", color=cv2.COLOR_BGR2YCrCb)
  Y = loadImages(I="trainGTs", color=cv2.COLOR_BGR2YCrCb)
  data = np.zeros((256, 256, 2))
  X = np.asarray(X)
  skinMask, skin = getSkin(X,Y)

  skinMask = np.asarray(skinMask)
  skinMask = skinMask[:,:,:,0]
  skinMask = skinMask.reshape((-1,))

  X = X.reshape(-1,3)
  
  for i in range(len(X)):
    Cr = X[i, 1]
    Cb = X[i, 2]
    if skinMask[i]:
      data[Cr, Cb, SKIN] +=1
    else:
      data[Cr, Cb, NOT_SKIN] += 1

  
  return data

def test(data, testImages, testGTs):
  X = loadImages(I="testImages", color=cv2.COLOR_BGR2YCrCb)
  X = np.asarray(X)
  s, h, w, c = X.shape
  X = X.reshape(-1,3)
  # Prediction
  P = np.zeros((s*h*w,))

  for i in range(len(X)):
    Cr = X[i, 1]
    Cb = X[i, 2]
    c_skin = 0
    c_back = 0
    try:
      c_skin = data[Cr,Cb,SKIN]
      c_back = data[Cr, Cb, NOT_SKIN]
    except:
      #do nothing
      pass
    if (c_skin > c_back):
      P[i] = SKIN
    else:
      P[i] = NOT_SKIN
  return P

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


def calcAccuracy2(Prediction, Truth):
  P = np.asarray(Prediction)
  T = np.asarray(Truth)
  
  miss = cv2.bitwise_xor(P, T)
  misses = np.sum(miss)
  total = T.size  # size of P, T or match
  loss = misses/total
  acc = 1 - loss
  print("Loss {:.2f}%".format(loss*100))
  print("Accuracy: {:.2f}%".format(acc*100))
  I = cv2.bitwise_and(P, T)
  U = cv2.bitwise_or(P, T)
  I = np.sum(I)
  U = np.sum(U)
  jacc = I/U
  print("Jaccard Index: {:.2f}%".format(jacc*100))
  return loss, acc, jacc

# import datasets


NOT_SKIN = 0
SKIN = 1
print ('Loading datasets...')
start = time.time()
trainImages = loadImages(I="trainImages",  color=cv2.COLOR_BGR2YCrCb)
trainGTs = loadImages(I="trainGTs", color=cv2.COLOR_BGR2YCrCb)
testImages = loadImages(I="testImages",  color=cv2.COLOR_BGR2YCrCb)
testGTs = loadImages(I="testGTs", color=cv2.COLOR_BGR2YCrCb)

end = time.time()
duration = end - start
print('Loaded in {:2f} s'.format(duration))
print("Training...")
start = time.time()
data = train()
end = time.time()
duration = end -start
print("Trained in {:.2f} s".format(duration))
np.save('./results/histogram-data.txt', data)
print ("Predicting...")
start = time.time()
predict = test(data, testImages, testGTs)
end = time.time()
duration = end - start
print("Predicted in {:.2f} s".format(duration))

X = loadImages(I="testImages")
Y = loadImages(I="testGTs")
skinMask, skin = getSkin(X, Y)
skinMask = np.asarray(skinMask)
skinMask = skinMask[:,:,:,0]
skinMask = skinMask.reshape(-1,)
s, h, w, c = np.asarray(Y).shape
# print (skinMask)
predict = np.asarray(predict, dtype=np.uint8)
predict[predict>0]=1


skinMask = skinMask.reshape((s,h,w))
predict = predict.reshape((s,h,w))

# showImages(skinMask, predict)
np.save("./results/bayesMask-2", predict)
calcAccuracy2(skinMask, predict)

#variação em que mede a probabilidade de um patch
