import numpy as np
import cv2
import glob
import os
import imutils
import math
import pdb
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

def reduceToContour(image, contour, hull=False, crop=False):

    if hull:
        contour = cv2.convexHull(contour)

    image = cv2.drawContours(image.copy(), [contour], 0, (255, 255, 255), -1)

    if crop:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        scale = 1.2  # cropping margin, 1 == no margin
        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle += 90
            rotated = True

        center = (int((x1+x2)/2), int((y1+y2)/2))
        size = (int(scale*(x2-x1)), int(scale*(y2-y1)))
        # # mostly for debugging purposes
        # cv2.circle(img_box, center, 10, (0, 255, 0), -1)

        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
        
        # !!!Important: which image to crop
        cropped = cv2.getRectSubPix(image.astype('float32'), size, center)
        cropped = cv2.warpAffine(cropped, M, size)

        croppedW = W if not rotated else H
        croppedH = H if not rotated else W
        
        image = cv2.getRectSubPix(
            cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))
    return image

#__main__


try:
  clf = joblib.load('quantitizer.pkl') 
  lut = np.load('lut.npy')
except:
  Xgt, G = getGTData()
  clf = createClassifier(Xgt)
  joblib.dump(clf, 'quantitizer.pkl') 
  lut = createLUT(Xgt, G)
  np.save('lut', lut)




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
  x = np.reshape(x, xshape)
  mask = np.zeros(xshape)
  mask[:,:,:]=0
  mask[y!=0]=(255,255,255)
  
  _, contours, _ = cv2.findContours(y, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

  for contour in contours:
    area = cv2.contourArea(contour)   
    
    # only contours large enough to contain object
    if area > int(.15*xh*xw):
      y = reduceToContour(mask, contour, hull=False, crop=False)
      result_image = x.copy()
      result_image[:, :] = (0, 0, 0)
      fg = (y>0)
      bg = np.invert(y>0)
      x = cv2.add((fg*x), (bg*result_image))
      filename = filename.replace('dataset', 'mask2')
      filename = filename.replace('.jpg', '-mask.jpg')
      directory = filename.split('/')
      directory = '/'.join(directory[:-1])
      if not os.path.exists(directory):    
          os.makedirs(directory)
      print ('w', filename)
      cv2.imwrite(filename, x)
      break



  


