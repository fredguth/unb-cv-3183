import numpy as np
import cv2

exp = input("Please enter from which experiment you want to measure: ")
fs_read = cv2.FileStorage(
    './exp-{}/Intrinsics.xml'.format(exp), cv2.FILE_STORAGE_READ)
intrinsic = fs_read.getNode('Intrinsics').mat()
fs_read.release()
fs_read = cv2.FileStorage(
    './exp-{}/Distortion.xml'.format(exp), cv2.FILE_STORAGE_READ)
distCoeff = fs_read.getNode('DistCoeffs').mat()
fs_read.release()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 15)
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
            

def drawLine (img, data, color):
    p1 = data["p1"]
    p2 = data["p2"]
    if (p2[0] > 0):
        cv2.line(img,tuple(p1),tuple(p2),color,2)
        print ("p1, p2: {}, {}".format(p1, p2))
        dist = np.linalg.norm(p2-p1)
        h, w, c = img.shape
        cv2.putText(img,"{} pixels".format(dist),(10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,1,cv2.LINE_AA)
        print("dist:{}".format(dist))           
    return img
    
while(capture.isOpened()):
    _, image = capture.read()    
    image = cv2.flip( image, 1)  # mirrors image
    h,  w = image.shape[:2]
    newcameraintrinsic, roi = cv2.getOptimalNewCameraMatrix(intrinsic,distCoeff,(w,h),1,(w,h))

    #undistort
    dst = cv2.undistort(image, intrinsic, distCoeff, None, newcameraintrinsic)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.setMouseCallback('Raw',mouse_callback, raw)
    cv2.setMouseCallback('Undistorted',mouse_callback, undistorted)

    
    image = drawLine(image, raw, (33,255,33))
    cv2.imshow('Raw', image)
    
    dst = drawLine(dst, undistorted, (255,33,255))
    cv2.imshow('Undistorted', dst)
    
    k = cv2.waitKey(60) & 0xFF
    if k==27:    # Esc key to stop
        break

capture.release()
cv2.destroyAllWindows()
