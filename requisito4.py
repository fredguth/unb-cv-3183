import numpy as np
import scipy.spatial as sp
import cv2
import time

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        data["mouseClicked"] = True
        frame = params['frame']
        bgr = frame[ line, column ]  
                 
        print("(line, column):({},{}); BGR ({})".format(line, column,bgr))
        params["bgr"] = bgr

def apply_red(frame, bgr, times):
    start = time.perf_counter()
    mask = create_mask(frame, bgr)
    inv_mask = np.invert (mask)
    lines, columns, channels = frame.shape
    result_image = np.zeros((lines, columns, channels), np.uint8)
    result_image[:,:]=(0,0,255)
    red = cv2.add((mask*frame), (inv_mask*result_image))
    end = time.perf_counter()
    times += [(end-start)*1000]
    cv2.imshow('red', red)
    cv2.moveWindow('red', 0, 300)

def create_mask (img, point):
    return (dist(img, point)>=13)

def dist(img, point):
    
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = point.reshape((1,3))
    distances = sp.distance.cdist(A,B, metric='euclidean')
    distances = distances.reshape((lines, columns, 1))
    return distances
    
    return distances

# ==============================================

video = cv2.VideoCapture(0)

ret, frame = video.read()
lines, columns, channels =  (frame.shape)
lines, columns, channels =  (frame.shape)
new_h= int(lines/2)
new_w= int(columns/2)

resized = cv2.resize(frame, (new_w, new_h)) 
count = 0
data = {
    'video': video,
    'frame': frame,
    'lines': new_h,
    'mouseClicked': False,
    'columns': new_w,
    'count': count,
    'times': [],
    'reading': []
}

while(True):
    count+=1
    if (count>=video.get(cv2.CAP_PROP_FRAME_COUNT)):
        video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        count = 1   
    start = time.perf_counter()
    success, frame = video.read()
    end = time.perf_counter()
    lines, columns, channels =  (frame.shape)
    new_h= int(lines/2)
    new_w= int(columns/2)
    resized = cv2.resize(frame, (new_w, new_h)) 
    data["reading"]+=[(end-start)*1000]
    data["frame"] = resized
    cv2.setMouseCallback('video', mouse_callback, data)
    if (data["mouseClicked"]):    
        apply_red(resized, data["bgr"], data["times"])
        
    cv2.imshow('video', resized)    
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        video.release()
        cv2.destroyAllWindows()
        print ("reading: {}".format(np.mean(data["reading"])))
        print ("masking: {}".format(np.mean(data["times"])))
        break


     



