import numpy as np
import cv2

intrinsic = np.load('Intrinsics.npy')
distCoeff = np.load('Distortion.npy')
capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FPS, 15)
capture.set(3, 640)
capture.set(4, 360)
cv2.namedWindow("Raw")
cv2.namedWindow("Undistorted")

raw = {
    "isMeasuring": False,
    "p1": np.asarray([-1, -1]),
    "p2": np.asarray([-1, -1])
}
undistorted = {
    "isMeasuring": False,
    "p1": np.asarray([-1, -1]),
    "p2": np.asarray([-1, -1])
}

def mouse_callback(event, column, line, flags, params):
    
    if event == 1: #left button in my mac
        params["isMeasuring"]= not params["isMeasuring"]
        if (params["isMeasuring"]):
            # first point
            params["p1"] = np.asarray([column, line])
            params["p2"] = np.asarray([-1, -1])       
        else:
            # second point
            p1 = params["p1"]
            p2 = np.asarray([column, line])
            params["p2"] = p2
            
            

while(capture.isOpened()):
    _, image = capture.read()    
    image = cv2.flip( image, 1)  # mirrors image
    h,  w = image.shape[:2]
    newcameraintrinsic, roi = cv2.getOptimalNewCameraMatrix(intrinsic,distCoeff,(w,h),1,(w,h))

    #undistort
    dst = cv2.undistort(image, intrinsic, distCoeff, None, newcameraintrinsic)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.setMouseCallback('Raw',mouse_callback, raw)
    cv2.setMouseCallback('Undistorted',mouse_callback, undistorted)

    # TODO: move to function DRY
    p1 = raw["p1"]
    p2 = raw["p2"]
    if (p2[0] > 0):
        cv2.line(image,tuple(p1),tuple(p2),(33,255,33),2)
        print ("p1, p2: {}, {}".format(p1, p2))
        dist = np.linalg.norm(p2-p1)
        h, w, c = image.shape
        cv2.putText(image,"{} pixels".format(dist),(10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(33,255,33),1,cv2.LINE_AA)
        print("dist:{}".format(dist))
    cv2.imshow('Raw', image)
    

    # p = undistorted["p1"]
    # p2 = undistorted["p2"]
    # if (p2[0] > 0):
    #     cv2.line(dst,tuple(p1),tuple(p2),(33,255,33),2)
    #     print ("p1, p2: {}, {}".format(p1, p2))
    #     dist = np.linalg.norm(p2-p1)
    #     h, w, c = dst.shape
    #     cv2.putText(dst,"{} pixels".format(dist),(10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(33,255,33),1,cv2.LINE_AA)
    #     print("dist:{}".format(dist))
    # cv2.imshow('Undistorted', dst)
    
    k = cv2.waitKey(60) & 0xFF
    if k==27:    # Esc key to stop
        break

capture.release()
cv2.destroyAllWindows()