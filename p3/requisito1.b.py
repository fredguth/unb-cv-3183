import numpy as np
from threading import Thread
import math
import scipy.spatial as sp
import cv2
import time

fs_read = cv2.FileStorage('disparity.xml', cv2.FILE_STORAGE_READ)
disp = fs_read.getNode('Disparity').mat()
fs_read.release()


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(
        image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def project3D(dispMatrix):
  disp = dispMatrix
  l, c = disp.shape
  # print (h,2)
  world = np.zeros((l, c, 3), np.float32)
  b = 120
  f = 25
  for h in range(0, l):
    for w in range(0, c):
      xL = w
      xR = w+disp[h, w]
      yL = h
      yR = h
      if not (xL-xR == 0):
        X = (b*(xL+xR))/(2*(xL-xR))
        Y = (b*(yL+yR))/(2*(xL-xR))
        Z = (b*f)/(xL-xR)
        world[h, w] = [X, Y, Z]
  return world
print('disp min', disp.min())
print('disp max', disp.max())
dmax = disp.max() - disp.min()
print ('disp max abs',  dmax)
print ('b * f/dmax', (120*25/dmax))
world = project3D(disp)
cv2.imwrite('disparity1.png', disp)
bsImg = (disp-disp.min())/disp.max()*255
bsImg = bsImg.astype(np.uint8)
cv2.imwrite('disparity2.png', bsImg)
# cv2.equalizeHist(bsImg, bsImg)
# cv2.imwrite('disparity2.png', bsImg)
depth = (world[:,:,2])
cv2.imwrite('depth1.png', depth)

depth = depth-depth.min()
print ('max depth', depth.max())
depth, _ = image_histogram_equalization(depth, number_bins=256)
depth = depth.astype(np.uint8)
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
mask = depth <= 230
inv_mask = depth >230
result_image = np.zeros(depth.shape, np.uint8)
result_image[:, :] = (0,0,0)
depth = cv2.add((mask*depth), (inv_mask*result_image))
cv2.imwrite('depth2.png', depth)
# print ('depth', depth)
# print ('histo', histo)
print ('2========')
print ('depth min', depth.min())
print('depth max', depth.max())
depth = (depth-depth.min())
print(depth[240:260, 240:260])
# depth = (depth/depth.max())
# depth = np.log(depth)
# depth = (depth-depth.min())
# depth = (depth/depth.max())
# depth = depth.astype(int)

# cv2.equalizeHist(depth, depth)
# depth = np.log(depth)
# print (depth[240:260,240:260 ])


cv2.imshow('basic', bsImg)
cv2.imshow('depth', depth)

while(True):

  k = cv2.waitKey(60) & 0xFF
  if k == 27:    # Esc key to stop
        break

cv2.destroyAllWindows()
