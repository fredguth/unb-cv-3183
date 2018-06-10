import numpy as np
import scipy.spatial as sp
import cv2
import time
import math


cv2.namedWindow("video")
cv2.moveWindow("video", 40, 40)
keepGoing = True
content = None
with open("./pd8-files/gtcar1.txt") as f:
    content = f.readlines()
bb = content[0].strip().split(',')
bx, by, a, b  =  (int(float(bb[0])), int(float(bb[1])), int(float(bb[2])), int(float(bb[3]))) 

bbox = (int(float(bb[0])), int(float(bb[1])),
        int(float(bb[2])), int(float(bb[3])))
bbox = (bx, by, a - bx, b - by)


frame = cv2.imread('./pd8-files/car1/00001.jpg', cv2.IMREAD_UNCHANGED)
template = frame[by:b, bx:a]


tracker = cv2.TrackerKCF_create()
ok = tracker.init(frame, bbox)

# cv2.imshow("template", template)
previous = bbox

def getIoU(b1, b2):
    boxA = np.asarray(b1).flatten()
    boxB = np.asarray(b2).flatten()

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

print ('\n\nUSE <SPACE> TO FORWARD')
while(keepGoing):
    f = 0
    mAcc = 0
    rbs = 1
    counter =1
    for i in range(945):
        
        fname = "{}.jpg".format(str(i+1).zfill(5))
        frame = cv2.imread('./pd8-files/car1/{}'.format(fname), cv2.IMREAD_UNCHANGED)
        
        c = content[i]
        c = c.strip().split(',')
        b1 = ((float(c[0]), float(c[1])), (float(c[2]), float(c[3])))
        # Update tracker
        ok, bbox = tracker.update(frame)
        if ok:   
                
            pt1 = (int(bbox[0]), int(bbox[1])) 
            pt2 = (int(bbox[0] + bbox[2]), int(bbox[1]+bbox[3]))
            
            frame = cv2.rectangle(frame, pt1, pt2, (0,0,0), 3)
            frame = cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
            
            previous = bbox    
            
            b2 = (pt1,pt2)
            
            if not math.isnan(b1[0][0]):
                acc = getIoU(b1, b2)
                mAcc = (mAcc*(counter-1)+acc)/counter
                rbs = math.exp(-30 * f/counter)
                print ('\nfalhas:', f)
                print ('frames válidos:', counter)
                counter = counter + 1
            
        else:
            
            # h, w, c = frame.shape
            # by, bx, bh, bw = previous
            # cv2.imshow('template', template)
            # res = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            # min_val, max_val, top_left, bottom_right = cv2.minMaxLoc(res)
            # by, bx = top_left
            # bbox = (by, bx, bh, bw)

            
            bb = content[i].strip().split(',')
            
            Nan = False
            for j in range(len(bb)):
                Nan = Nan or math.isnan(float(bb[j]))
            if not Nan:
                bx, by, a, b = (int(float(bb[0])), int(float(bb[1])),
                                int(float(bb[2])), int(float(bb[3])))
                mAcc = (mAcc*(counter-1))/counter
                f = f+1
                rbs = math.exp(-30* f/counter)
                counter = counter + 1
                bbox = (int(float(bb[0])), int(float(bb[1])),
                        int(float(bb[2])), int(float(bb[3])))
                bbox = (bx, by, a - bx, b - by)
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)

                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[0] + bbox[2]), int(bbox[1]+bbox[3]))
                frame = cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 2)
                
        cv2.putText(frame, "f:{}, acc: {:.2f}, rbs: {:.2f}".format(i, mAcc*100, rbs*100), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow('video', frame)
        
        while (keepGoing):
            k = cv2.waitKey(33) & 0xff

            if k == 27:
                cv2.destroyAllWindows
                keepGoing = False
                break
            if k == 32:  # spae
                if (i==944):
                    keepGoing = False
                break
   
while (True):
    cv2.imshow('video', frame)
    k = cv2.waitKey(33) & 0xff

    if k == 27:
        cv2.destroyAllWindows
        
        break
    


     



