import numpy as np
import scipy.spatial as sp
import cv2


img = cv2.imread('./media/venn.png',cv2.IMREAD_UNCHANGED)
imgType = type(img[0,0]).__name__
data = {
    'bgr': None,
    'lum': None,
    'mouseclicked': False
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
        params["mouseclicked"]=True
        if (imgType == 'uint8'): #grayscale
            lum = img [line, column]
            params["lum"] = lum
            print("(line, column):({},{}); Luminosity ({})".format(line, column,lum))
        elif (imgType == 'ndarray'): #color         
            bgr = img [ line, column ] 
            params["bgr"] = bgr          
            print("(line, column):({},{}); BGR ({})".format(line, column,bgr))

def create_mask (img, point):
    return (dist(img, point)>=13)

def apply_red (img, point):
    
    mask = create_mask(img, point)
    inv_mask = np.invert (mask)
    if (imgType == "uint8"): #grayscale
        img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR) 
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
        color = data["bgr"] if (imgType == 'ndarray') else data["lum"]
        apply_red(img, color)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

