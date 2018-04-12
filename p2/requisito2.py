import numpy as np
import cv2


video = cv2.VideoCapture(0)
# set display at 640x360
video.set(3, 640)
video.set(4, 360)


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

def snapshot(image):
    print (image)

while(1):
    ret, frame = video.read()

    if ret==True:
        cv2.imshow('stream',frame)
        # cv2.setMouseCallback('stream',mouse_callback, data)
        k = cv2.waitKey(2) & 0xFF
        if k==27:    # Esc key to stop
            break
        if k==32:    # Espace bar to snapshot
            snapshot(frame)
    else:
        break

video.release()
cv2.destroyAllWindows()
