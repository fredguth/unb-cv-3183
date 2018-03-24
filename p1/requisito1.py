import numpy as np
import scipy.spatial as sp
import cv2

global img
img = cv2.imread('./media/venn.jpg',cv2.IMREAD_UNCHANGED)
imgType = type(img[0,0]).__name__

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        if (imgType == 'uint8'): #grayscale
            print (img [line, column])
        elif (imgType == 'ndarray'): #color         
            bgr = img [ line, column ]           
            print("(line, column):({},{}); BGR ({})".format(line, column,bgr))

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

