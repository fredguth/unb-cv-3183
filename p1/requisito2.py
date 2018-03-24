import numpy as np
import scipy.spatial as sp
import cv2


img = cv2.imread('./media/venn.png',cv2.IMREAD_COLOR)
imgType = type(img[0,0]).__name__
data = {
    'bgr': None,
    'mouseclicked': False
}
def dist(img, point):
 
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = point.reshape((1,3))
    distances = sp.distance.cdist(A,B, metric='euclidean')
    distances = distances.reshape((lines, columns, 1))
    return distances

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        params["mouseclicked"]=True
        if (imgType == 'uint8'): #grayscale
            print (img [line, column])
        elif (imgType == 'ndarray'): #color         
            bgr = img [ line, column ] 
            params["bgr"] = bgr          
            print("(line, column):({},{}); BGR ({})".format(line, column,bgr))

def create_mask (img, point):
    return (dist(img, point)>=13)

def apply_red (img, bgr):
    mask = create_mask(img, bgr)
    inv_mask = np.invert (mask)
    lines, columns, channels = img.shape
    result_image = np.zeros((lines, columns, channels), np.uint8)
    result_image[:,:]=(0,0,255)
    red = cv2.add((mask*img), (inv_mask*result_image))
    cv2.imshow('red', red)
    cv2.moveWindow('red', columns, 45)

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback, data)
    if (data["mouseclicked"]):
        apply_red(img, data["bgr"])
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

