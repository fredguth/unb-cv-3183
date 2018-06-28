import numpy as np
import cv2
import glob
import os
import imutils
import math
import scipy.spatial as sp
from sklearn.cluster import MiniBatchKMeans

filenames = glob.glob('./quant/*/*')
for filename in filenames:

  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:,:,0]
  ret, thresh = cv2.threshold(gray.copy(), 130, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  for contour in contours:
    area = cv2.contourArea(contour)    
    # only contours large enough to contain object
    if area > 300000:
      dummy = thresh
      dummy[:,:,:]=0
      contour = cv2.drawContours(dummy, [contour], 0, (255, 255, 255), -1)
      break
  mask = 255 -contour
  file = filename
  file = file.replace('.jpg', '-wfor.jpg')
  cv2.imwrite(file, mask)
  file = file.replace('wfor', 'bfor')
  cv2.imwrite(file, contour)