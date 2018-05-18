import numpy as np
import scipy.spatial as sp
import cv2
import glob
import matplotlib.pyplot as plt
import time
from sklearn import svm



# return arrays of images ORI, GT
def loadImages(I, color=cv2.COLOR_BGR2RGB):
  Iout = []
  imageFolder = './sfa/ORI/'
  truthFolder = './sfa/GT/'
  training = range(1, 783)
  testing = range(951, 1119)
  if   (I == "trainImages"):
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
    print ("Not valid dataset")
  
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
    Xtrain = loadImages(I=trainImages, color=color)
    Ytrain = loadImages(I=trainGTs, color=color)

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
def train(X, Y):

  data = np.zeros((256, 256, 2))
  X = np.asarray(X)
  skinMask, skin = getSkin(X,Y)

  skinMask = np.asarray(skinMask)
  skinMask = skinMask[:,:,:,0]
  skinMask = skinMask.reshape((-1,))

  X = X.reshape(-1,3)
  
  for i in range(len(X)):
    Cr = X[i, 1]
    Cb = X[i,2]
    if skinMask[i]:
      data[Cr, Cb, SKIN] +=1
    else:
      data[Cr, Cb, NOT_SKIN] += 1

  
  return data

# def testKnn(data, testImages):
#   X = loadImages(I=testImages, color=cv2.COLOR_BGR2YCrCb)
#   X = np.asarray(X)
#   s, h, w, c = X.shape
#   X = X.reshape(-1,3)
#   X = 
#   # Prediction
#   P = np.zeros((s*h*w,))

#   for i in range(len(X)):
#     Cr = X[i, 1]
#     Cb = X[i, 2]
#     c_skin = 0
#     c_back = 0
#     try:
#       c_skin = data[Cr,Cb,SKIN]
#       c_back = data[Cr, Cb, NOT_SKIN]
#     except:
#       #do nothing
#       pass
#     if (c_skin > c_back):
#       P[i] = SKIN
#     else:
#       P[i] = NOT_SKIN
#   return P

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
  I = np.sum(cv2.bitwise_and(P, T))
  U = np.sum(P)+np.sum(T)-I
  # U = cv2.bitwise_or (P,T)
  # I = np.sum(I)
  # U = np.sum(U)
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
  # I = cv2.bitwise_and(P, T)
  # U = cv2.bitwise_or(P, T)
  # I = np.sum(I)
  # U = np.sum(U)
  I = np.sum(cv2.bitwise_and(P, T))
  U = np.sum(P)+np.sum(T)-I
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
end = time.time()
duration = end - start
print ('Loaded in {:2f} s'.format(duration))
print ("Training...")
start = time.time()
data = train(trainImages, trainGTs)



from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
mask  = np.where((data[:,:,SKIN]>0)|(data[:,:,NOT_SKIN]>0))
X = np.stack(mask, axis=-1)
Y = data[:, :, SKIN]>data[:,:,NOT_SKIN]
Y = Y[mask]

h = .05  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = neighbors.KNeighborsClassifier(3)

clf.fit(X, Y)

end = time.time()
duration = end-start
np.save('./results/knn-data.txt', data)
print("Trained in {:.2f} s".format(duration))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# x_min, x_max = 0, 255
# y_min, y_max = 0, 255
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
                    
print ("Predicting...")
start = time.time()
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
T = loadImages(I="testImages", color=cv2.COLOR_BGR2YCrCb)
T = np.asarray(T)
T = T.reshape(-1, 3)
W = T[:,1:3]

W = np.unique (W, axis=0)

Z = clf.predict(W)


print("Post processing...")
np.save('./results/knn-prediction.txt', Z)

s, c = T.shape
predictMask = np.zeros(s)
for i in range(len(T)):
  color = T[i]
  CrCb = color[1:3]
  Cr = color[1]
  Cb = color[2]
  j = np.where((W[:, 0] == Cr) & (W[:, 1] == Cb))
  predictMask[i] = Z[j]

U = loadImages(I="testGTs")
U = np.asarray(U)
U = U[:,:,:,0]
U = U.reshape((-1,))
mask = np.where(U>0)
U = np.zeros((U.shape))
U[mask]=1

end = time.time()
duration = end-start
print("Predicted in {:.2f} s".format(duration))
calcAccuracy2(predictMask, U)
np.save('./results/knnMask-2', predictMask)


# # Put the result into a color plot
# # Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
#             edgecolor='None', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("K-nn (k = 15, weights = distance)")
# print('saving figure')
# plt.savefig('./results/knn-15.png')
# plt.show()
# print('Finished')
