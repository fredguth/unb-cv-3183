import numpy as np
from threading import Thread
import math
import scipy.spatial as sp
import cv2
import time

imgL = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeL.png', cv2.IMREAD_GRAYSCALE))
imgR = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeR.png', cv2.IMREAD_GRAYSCALE))

window_size = 3
min_disp = 0
max_disp = 112

def normalize(matrix):
  mat = matrix
  min_mat = mat.min()
  max_mat = mat.max()
  mat = mat - min_mat
  mat = mat/max_mat
  return mat

def computeDisparity(imgL, imgR, k, maxshift):
  start = time.time()
  print ("staring at {}".format(start))
  disp = np.zeros(imgL.shape) 
  b = int (k/2)
  l, c = imgL.shape
  
  for h in range (b, l-b-1):
    for w in range(b, c-b-1):
      template = imgL[h-b:h+b+1, w-b:w+b+1]
      disparity = 0
      cost = math.inf
      for shift in range(0, maxshift):
        window = imgR[h-b:h+b+1, (w+shift)-b:(w+shift)+b+1]
        if (window.shape == template.shape):
          sad = (abs(template-window)).sum()
          if sad < cost:
            cost = sad
            disparity = shift
        
      print("H:{}, W:{}".format(h, w))
      disp[h, w] = disparity
  end = time.time()
  print ("{}min {}secs later...".format(int((end-start)/60), int((end-start)%60)))
  return disp

def project3D(dispMatrix):
  disp = dispMatrix
  l, c = disp.shape
  # print (h,2)
  world = np.zeros((l, c, 3), np.float32)
  b = 120
  f = 25
  for h in range(0,l):
    for w in range (0,c):
      xL = w
      xR = w+disp[h,w]
      yL = h
      yR = h 
      if not (xL-xR==0):     
        X = (b*(xL+xR))/(2*(xL-xR))
        Y = (b*(yL+yR))/(2*(xL-xR))
        Z = (b*f)/(xL-xR)
        world[h,w] = [X, Y, Z]
  return world
# stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
#                         numDisparities=max_disp,
#                         blockSize=window_size,
#                         P1=8*3*window_size**2,
#                         P2=32*3*window_size**2,
#                         disp12MaxDiff=1,
#                         uniquenessRatio=10,
#                         speckleWindowSize=100,
#                         speckleRange=32
#                         )
stereo = cv2.StereoBM_create(
                             numDisparities=max_disp
                             ,
                             blockSize=window_size
                             )
print ('computing disparity...')
# disp = stereo.compute(imgL, imgR)
disp = computeDisparity(imgL, imgR, window_size, max_disp)
cv2.imwrite('disparity.png', disp)
bsImg = (disp-disp.min())/disp.max()

world = project3D(disp)
depth = world[:,:,2]
depth = (depth-depth.min())
print ('depth min', depth.min())
print('depth max', depth.max())
depth = (depth/depth.max())


cv2.imshow('basic', bsImg)
cv2.imshow('depth', depth)

while(True):

  k = cv2.waitKey(60) & 0xFF
  if k == 27:    # Esc key to stop
        break

cv2.destroyAllWindows()
