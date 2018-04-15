import numpy as np
import cv2
import os

fs_read = cv2.FileStorage('Intrinsics.xml', cv2.FILE_STORAGE_READ)
intrinsic = fs_read.getNode('Intrinsics').mat()
fs_read.release()
fs_read = cv2.FileStorage('Distortion.xml', cv2.FILE_STORAGE_READ)
distCoeff = fs_read.getNode('DistCoeffs').mat()
fs_read.release()

board_w = 8  # horizontal enclosed corners on chessboard
board_h = 6  # vertical enclosed corners on chessboard

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


exp = input("Please enter experiment number: ")
cwd = os.getcwd()
directory = cwd + '/exp-'+exp
if not os.path.exists(directory):
    os.makedirs(directory)
square = float(input("Please enter size of a chessboard square in mm:"))
count = 0

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 15)
capture.set(3, 640)
capture.set(4, 360)
cv2.namedWindow("Raw")
cv2.namedWindow("Undistorted")
cv2.namedWindow("Chess")

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
    
def calculateExtrinsics(image, exp, count):
    object_points = np.zeros((board_h*board_w, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)*square

    print('object_points')

    found, corners = cv2.findChessboardCorners(
        image, (board_w, board_h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)
    # If found corners, refine
    if found == True:
        corners = cv2.cornerSubPix(
            image, corners, (11, 11), (-1, -1), criteria)
    
        # cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]])
        ret, r, t, inliners = cv2.solvePnPRansac(object_points, corners, intrinsic, distCoeff)
        R, j = cv2.Rodrigues(r)
        print ('R:', R)
        print ('t:', t)

    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.drawChessboardCorners(
         image, (board_w, board_h), corners, found)
    cv2.imshow('Chess', image)

while(capture.isOpened()):
    _, image = capture.read()    
    image = cv2.flip( image, 1)  # mirrors image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,  w = image.shape[:2]
    newcameraintrinsic, roi = cv2.getOptimalNewCameraMatrix(intrinsic,distCoeff,(w,h),1,(w,h))

    #undistort
    dst = cv2.undistort(image, intrinsic, distCoeff, None, newcameraintrinsic)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.setMouseCallback('Raw',mouse_callback, raw)
    cv2.setMouseCallback('Undistorted',mouse_callback, undistorted)

    
    image = drawLine(image, raw, (33,255,33))
    cv2.imshow('Raw', image)
    
    dst = drawLine(dst, undistorted, (255,33,255))
    cv2.imshow('Undistorted', dst)
    
    k = cv2.waitKey(60) & 0xFF
    if k==27:    # Esc key to stop
        break
    if k == 32:  # Space -> snapshot
        count += 1
        filename = directory + '/snap-{}-{}.png'.format(exp, count)
        print(filename)
        cv2.imwrite(filename, dst)
        calculateExtrinsics(dst, exp, count)

capture.release()
cv2.destroyAllWindows()

t = -R 
