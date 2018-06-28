import numpy as np
import cv2
import glob
import os
import imutils
import math
import scipy.spatial as sp
from sklearn.cluster import MiniBatchKMeans

def quantitize(img, n):

  (h, w, c) = (np.asarray(img)).shape

  img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
  img = np.reshape(img, (h*w, 3))  
  
  clt = MiniBatchKMeans(n_clusters=n)
  colors = clt.fit_predict(img)
  
  quant = clt.cluster_centers_.astype("uint8")[colors]
  quant = np.reshape(quant, (h, w, 3))
  quant = cv2.cvtColor(quant, cv2.COLOR_YCrCb2BGR)
  
  return quant


filenames = glob.glob('./dataset/*/*')

for filename in filenames:
  image = cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR),(448,448))
  n = 6
  image = quantitize(image, n)
  name = filename
  name = name.replace('dataset', 'quant')
  name = name.replace('.jpg', '-{}.jpg'.format(n))
  directory = name.split('/')
  directory = '/'.join(directory[:-1])
  if not os.path.exists(directory):    
      os.makedirs(directory)
  print ('w ', name)
  cv2.imwrite(name, image)

  