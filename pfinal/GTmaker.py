import numpy as np
import cv2
import glob
import os
import imutils
import math
import scipy.spatial as sp
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from sklearn.externals import joblib

def createClassifier (images):
  n_colors = 64
  n, h, w, d = original_shape = tuple(images.shape)
  assert d == 3
  image_array = np.reshape(images, ( n*h*w, d))

  print("Fitting model on a small sub-sample of the data")
  image_array_sample = shuffle(image_array, random_state=0)[:n*1000]
  clf = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
  print ("Model fit complete")
  return clf

def getGTData():
  # Xgt = original source data of groundtruth
  # G = GTruth
  GTfiles = glob.glob('./GT/*/*')
  Xgt = []
  G = []
  for filename in GTfiles:
    print ('r', filename)
    xgFile = filename.replace('GT', 'dataset')
    xgFile = xgFile.replace('-mask', '')
    gtImage = cv2.imread(filename)
    xgImage = cv2.imread(xgFile)
    xgImage = cv2.resize(xgImage, (448,448))
    Xgt.append(xgImage)
    G.append(gtImage)
  Xgt =np.asarray(Xgt)
  G = np.asarray(G)
  return Xgt, G

# LUT is a look up table that says for each 64 colors, which are most 
# likely foreground and which are most likely background
def createLUT(Xgt, G):
  n, h, w, d = original_shape = tuple(Xgt.shape)
  Xgt= np.reshape(Xgt, ( n*h*w, d))
  labels = clf.predict(Xgt)
  gt = G/255
  gt = gt[:,:,:,0]
  gn, hn, wn= gt.shape
  gt = np.reshape(gt,(gn*hn*wn,))

  #foreground == 1
  #background == 0
  lut = np.zeros((64, 2))
  for i in range(len(labels)):
    label = labels[i]
    if gt[i]==0: # background
      lut[label, 0] +=1
    else: # foreground
      lut[label, 1] += 1

  lut =  (lut[:,1]>lut[:,0]).astype("uint8")
  return lut

#__main__
Xgt, G = getGTData()
clf = createClassifier(Xgt)
joblib.dump(clf, 'quantitizer.pkl') 
lut = createLUT(Xgt, G)


filenames = glob.glob('./dataset/*/*')

for filename in filenames:
  
  x = cv2.imread(filename)
  x = cv2.resize(x, (448,448))
  (xh, xw, xd) = xshape = tuple(x.shape)
  x = np.reshape(x, (xh*xw, xd))
  xl = clf.predict(x)
  y = np.zeros(x.shape)
  y = lut[xl]*255
  y = np.reshape(y, (xh, xw))
  
  filename = filename.replace('dataset', 'mask')
  filename = filename.replace('.jpg', '-mask.jpg')
  directory = filename.split('/')
  directory = '/'.join(directory[:-1])
  if not os.path.exists(directory):    
      os.makedirs(directory)
  print ('w', filename)
  cv2.imwrite(filename, y)


