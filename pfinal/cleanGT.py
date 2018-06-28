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

GTfiles = glob.glob('./GT/*/*')
  
for filename in GTfiles:
  g = cv2.imread(filename)
  gray = cv2.cvtColor(g, cv2.COLOR_BGR2YCrCb)
  gray = gray[:,:,0]
  ret, thresh = cv2.threshold(gray.copy(), 1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  cv2.imwrite(filename, thresh)