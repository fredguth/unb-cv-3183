import numpy as np
import cv2


img = cv2.imread('./media/baboon.jpg', cv2.IMREAD_COLOR)
data = {
    "isMeasuring": False,
    "p1":[],
    "p2":[]
}
def dist(img, point):
    channels = 1
    if (imgType == "ndarray"): #color imag
        lines, columns, channels = img.shape #channels = 3
    else:
        lines, columns = img.shape
    A = img.reshape((lines*columns, channels))
    # B = point if (imgType == 'uint8') else point.reshape((1,3))
    B = point.reshape((1,channels))
    distances = sp.distance.cdist(A,B, metric='euclidean')
    distances = distances.reshape((lines, columns, 1))
    return distances

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        params["isMeasuring"]= not params["isMeasuring"]
        if (params["isMeasuring"]):
            # first point
            params["p1"] = [line, column]
            print ("p1: {}".format(params["p1"]))
        else:
            # second point
            params["p2"] = [line, column]
            print ("p2: {}".format(params["p2"]))

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback, data)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

