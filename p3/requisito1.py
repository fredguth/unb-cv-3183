import numpy as np
from threading import Thread
import scipy.spatial as sp
import cv2
import time

imgL = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeL.png', cv2.IMREAD_COLOR))
imgR = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeR.png', cv2.IMREAD_COLOR))

WINDOW_SIZE = 5
MIN_DISP = 16
MAX_DISP = 96
THRESHOLD = 256
def normalize(matrix):
  mat = matrix
  min_mat = mat.min()
  max_mat = mat.max()
  mat = mat - min_mat
  mat = mat/max_mat
  return mat


def scanLine(line, imgL, imgR, window_size, width, height):
    disp = np.zeros((width-window_size), dtype=np.uint8) 
    if line % 50 == 0:
        print('{}% done'.format(int((line / (height - window_size)) * 100)))

    for column in range(width - window_size):
        template = imgL[line:(line + window_size), column:(column + window_size)]
        start = column - MAX_DISP
        if start < 0:
            start = 0
        end = column + window_size
        strip = imgR[line:(line + window_size), start:end]

        match = cv2.matchTemplate(strip, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
        disparity = (column - start) - min_loc[0]
        # if (min_val/window_size**2 < THRESHOLD):
        if (min_val/WINDOW_SIZE**2 < THRESHOLD) and (disparity > MIN_DISP):
          disp[column] = disparity
   
    return disp
    
def computeDisparity(imgL, imgR, k, maxshift):
  start = time.time()
  l, c, _ = imgL.shape
  disp = np.zeros((l,c), np.uint8)
  for h in range (l-WINDOW_SIZE):
    disp[h, WINDOW_SIZE:c] = scanLine(h, imgL, imgR, WINDOW_SIZE, c, l)
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
 
# stereoMIN_DISPmin_disp,
#                         numDisparities=MAX_DISP,
#                         blockSize=WINDOW_SIZE,
#                         P1=8*3*WINDOW_SIZE**2,
#                         P2=32*3*WINDOW_SIZE**2,
#                         disp12MaxDiff=1,
#                         uniquenessRatio=10,
#                         speckleWindowSize=100,
#                         speckleRange=32
#                         )
# stereo = cv2.StereoBM_create(
#                              numDisparities=MAX_DISP
#                              ,
#                              blockSize=WINDOW_SIZE
#                              )
print ('computing disparity...')
# disp = stereo.compute(imgL, imgR)
disp = computeDisparity(imgL, imgR, WINDOW_SIZE, MAX_DISP)
fs_write = cv2.FileStorage(
    'disparity.xml', cv2.FILE_STORAGE_WRITE)
fs_write.write('Disparity', disp)
fs_write.release()

