import numpy as np
import scipy.spatial as sp
import cv2

global img
global called
called = 0
# img = cv2.imread('dummy.png',cv2.IMREAD_UNCHANGED)
img = cv2.imread('venn.png',cv2.IMREAD_COLOR)
imgType = type(img[0,0]).__name__

def dist(img, point):
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = np.zeros(shape=(1,3))
    B[0]=point
    distances = sp.distance.cdist(A,B, metric='euclidean')
    return distances
     

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        if (imgType == 'uint8'): #grayscale
            print (img [line, column])
        elif (imgType == 'ndarray'): #color         
            bgr = img [ line, column ]           
            print("(line, column):({},{}); BGR ({})".format(line, column,bgr))
            print (params)
            create_mask(img, bgr)

def create_mask (img, point):
    distances = dist(img, point)
    lines, columns, channels = img.shape
    mask = distances.reshape((lines, columns, 1))
    mask = (mask<13).astype(int) * [0,0,255]
    cv2.imwrite('mask.png', mask)

while(1):
    cv2.imshow('img',img)
    cv2.setMouseCallback('img',mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

