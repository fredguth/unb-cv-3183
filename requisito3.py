import numpy as np
import scipy.spatial as sp
import cv2


def dist(img, point):
    # TODO: test np.hypot
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = np.zeros(shape=(1,3))
    B[0]=point
    distances = sp.distance.cdist(A,B, metric='euclidean')
    distances = distances.reshape((lines, columns, 1))
    return distances
     

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac

        bgr = frame[ line, column ]           
        print("(line, column):({},{}); BGR ({})".format(line, column,bgr))
        # mask = create_mask(frame, bgr)
        # inv_mask = np.invert (mask)
        # lines, columns, channels = frame.shape
        # result_image = np.zeros((lines, columns, channels), np.uint8)
        # result_image[:,:]=(0,0,255)
        # red = cv2.add((mask*frame), (inv_mask*result_image))
        # cv2.imshow('red', red)
        # cv2.moveWindow('red', columns, 45)
def create_mask (img, point):
    return (dist(img, point)>=13)

global video
global frame
video = cv2.VideoCapture('coralastra2.avi')
ret, frame = video.read()
lines, columns, channels =  (frame.shape)
count = 0
while(True):
    count+=1
    if (count>=video.get(cv2.CAP_PROP_FRAME_COUNT)):
        video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        count = 1    
    success, frame = video.read()
    cv2.imshow('video', frame)    
    
    # cv2.setMouseCallback('video', mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        video.release()
        cv2.destroyAllWindows()
        break

