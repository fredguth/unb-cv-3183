import numpy as np
import scipy.spatial as sp
import cv2

global img
img = cv2.imread('../pfinal/dataset/72075/72075-00009.jpg',cv2.IMREAD_COLOR)
imgType = type(img[0,0]).__name__

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            bgr = img [ line, column ]   
            rgb = list(reversed(bgr))        
            print("(line, column):({},{}); BGR ({}); RGB ({})".format(line, column,bgr, rgb))

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

