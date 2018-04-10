import numpy as np
import cv2


img = cv2.imread('./media/baboon.jpg', cv2.IMREAD_COLOR)
data = {
    "isMeasuring": False,
    "p1":[],
    "p2":[],
}

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        params["isMeasuring"]= not params["isMeasuring"]
        if (params["isMeasuring"]):
            # first point
            global img
            img = cv2.imread('./media/baboon.jpg', cv2.IMREAD_COLOR)
            params["p1"] = np.asarray([column, line])
            print ("p1: {}".format(params["p1"]))
        else:
            # second point
            p1 = params["p1"]
            p2 = np.asarray([column, line])
            params["p2"] = p2
            params["line"] = cv2.line(img,tuple(p1),tuple(p2),(255,0,0),3)
            print ("p2: {}".format(p2))
            dist = np.linalg.norm(p2-p1)
            print("dist:{}".format(dist))

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback, data)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

