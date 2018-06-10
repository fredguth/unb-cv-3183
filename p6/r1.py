import numpy as np
import scipy.spatial as sp
import cv2
import time


cv2.namedWindow("video")

keepGoing = True
content = None
with open("./pd8-files/gtcar1.txt") as f:
    content = f.readlines()
previous = []
while(keepGoing):
    # for i in range (9928):
    for i in range(945):
        fname = "{}.jpg".format(str(i+1).zfill(5))
        frame = cv2.imread('./pd8-files/car1/{}'.format(fname), cv2.IMREAD_UNCHANGED)
        # # ###############################

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if (previous != []):
        #     frame = cv2.absdiff(frame, previous)
        # previous = frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # ######################################
        try:
            bb = content[i].strip().split(',')
            pt1 = (int(float(bb[0])), int(float(bb[1]))) 
            pt2 = (int(float(bb[2])), int(float(bb[3]))) 
            # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img
            frame = cv2.rectangle(frame, pt1, pt2, (0,0,0), 3)
            frame = cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
        except:
            pass
        cv2.imshow('video', frame)
        
        while (keepGoing):
            k = cv2.waitKey(33) &0xff
            
            if k==27:
                cv2.destroyAllWindows
                keepGoing = False
                break
            if k==32: #spae
                break
    

     



