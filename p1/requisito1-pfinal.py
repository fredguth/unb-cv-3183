import numpy as np
import scipy.spatial as sp
import cv2

global img
img = cv2.imread('../pfinal/dataset/72075/72075-00014.jpg',cv2.IMREAD_COLOR)
imgType = type(img[0,0]).__name__

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
            cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            color = img [ line, column ]   
            print("(line, column):({},{}); YCrCb ({}); ".format(line, column,color))

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

