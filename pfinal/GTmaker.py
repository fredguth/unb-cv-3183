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

def findClusters (images):
  n_colors = 64

  # Convert to floats instead of the default 8 bits integer coding. Dividing by
  # 255 is important so that plt.imshow behaves works well on float data (need to
  # be in the range [0-1])
  images = np.array(images, dtype=np.float64) / 255

  n, h, w, d = original_shape = tuple(images.shape)
  assert d == 3
  image_array = np.reshape(images, ( n*h*w, d))

  print("Fitting model on a small sub-sample of the data")
  image_array_sample = shuffle(image_array, random_state=0)[:1000]
  kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
  print ("Model fit complete")
  return kmeans

def getAllColors ():
  table = []
  for B in range(256):
    for G in range (256):
      for R in range (256):
        table.append((B,G,R))
  table = np.asarray(table)
  return table

def getGTData()
  # Xgt = original source data of groundtruth
  # G = GTruth
  GTfiles = glob.glob('./GT/*/*')
  Xgt = []
  G = []
  for filename in GTfiles:
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

def makeColormap(X)
  allColors = getAllColors()
  kmeans = findClusters(X)
  labels = kmeans.predict(allColors)
  cc = kmeans.cluster_centers_

  colormap = allColors
  for i in range(len(labels)):
    colormap[i] = cc[labels[i]]
  colormap = np.reshape(colormap, (256, 256, 256, 3))
  
  return colormap

#__main__
Xgt, G = getGTData()
colormap = makeColormap(Xgt)
colormap = np.save('colormap', colormap)