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


def scan_line(line, imgL, imgR, window_size, width, height):
    disp = np.zeros((width-window_size), dtype=np.float16) #unint8?
    if line % 20 == 0:
        print('{}% done'.format((line / (height - window_size)) * 100))

    for column in range(width - window_size):
        template = imgL[line:(line + window_size), column:(column + window_size)]
        start = column - max_disp
        if start < 0:
            start = 0
        end = column + window_size
        strip = imgR[line:(line + window_size), start:end]

        match = cv2.matchTemplate(strip, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

        disp[column] = (column - start) - min_loc[0]
    #     x_r = strip_start + min_loc[0]
    #     if (min_val/window_size**2 < THRESHOLD) and (column - x_r) > 10:  # only OK matches
    #         # X = (BASE_LINE * (column + x_r))/(2 * (column - x_r))
    #         # Y = (BASE_LINE * (2 * line))/(2 * (column - x_r))
    #         Z = (BASE_LINE * FOCAL_LENGTH)/(column - x_r)
    #         depth_vet[x_l] = Z
    # return depth_vet
    return disp
    
def computeDisparity(imgL, imgR, k, maxshift):
  start = time.time()
  print ("staring at {}".format(start))
  disp = np.zeros(imgL.shape) 
  l, c = imgL.shape
  for h in range (l-window_size):
    disp[h, window_size:c] = scan_line(h, imgL, imgR, window_size, c, l)
    # for w in range(b, c-b-1):
    #   template = imgL[h-b:h+b+1, w-b:w+b+1]
    #   disparity = 0
    #   cost = math.inf
    #   for shift in range(0, maxshift):
    #     window = imgR[h-b:h+b+1, (w+shift)-b:(w+shift)+b+1]
    #     if (window.shape == template.shape):
    #       sad = (abs(template-window)).sum()
    #       if sad < cost:
    #         cost = sad
    #         disparity = shift
        
      # print("H:{}, W:{}".format(h, w))
      # disp[h, w] = disparity
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
