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

def naive(frame, bgr, times):
    start = time.perf_counter()
    lines, columns, channels = frame.shape
    print (lines, columns)
    red = frame
    B1, G1, R1 = bgr
    D=np.ndarray(shape=(lines, columns))
    for i in range (0, lines):
        for j in range (0, columns):
            B2, G2, R2 = frame[i,j]
            dist = ((B2-B1)**2+(G2-G1)**2+(R2-R1)**2)**(0.5)
            if (dist<13):
                red[i,j]=(0,0,255)
    end = time.perf_counter()
    print ((end-start)*1000)
    times += [(end-start)*1000]
    cv2.imshow('red', red)
    cv2.moveWindow('red', lines+100, columns+300)



# ==============================================

video = cv2.VideoCapture('./media/monicatoy.mp4')

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
       naive(resized, data["bgr"], data["times"])
        
    cv2.imshow('video', resized)    
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        video.release()
        cv2.destroyAllWindows()
        print ("reading: {}".format(np.mean(data["reading"])))
        print ("masking: {}".format(np.mean(data["times"])))
        break


     



