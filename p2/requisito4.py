import numpy as np
import pandas as pd
import cv2
import os


board_w = 8  # horizontal enclosed corners on chessboard
board_h = 6  # vertical enclosed corners on chessboard
square = 2.74

object_points = np.zeros((board_h*board_w, 3), np.float32)
object_points[:, :2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1, 2)*square

R = None
t = None
distance = 0


fs_read = cv2.FileStorage('./exp-0/Intrinsics.xml', cv2.FILE_STORAGE_READ)
intrinsic = fs_read.getNode('Intrinsics').mat()
fs_read.release()
fs_read = cv2.FileStorage('./exp-0/Distortion.xml', cv2.FILE_STORAGE_READ)
distCoeff = fs_read.getNode('DistCoeffs').mat()
fs_read.release()
count = 0

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 15)
capture.set(3, 640)
capture.set(4, 360)
cv2.namedWindow('Raw')
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

saving = False
goodResult = False
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
        p1_3D = project3D(p1)
        p2_3D = project3D(p2)
        dist = np.linalg.norm(p2-p1)
        dist3D = np.linalg.norm(p2_3D-p1_3D)
        
        h, w= img.shape
        cv2.putText(img,"{} pixels".format(dist),(10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,1,cv2.LINE_AA)
        cv2.putText(img, "{} cm".format(dist3D), (10, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        print("dist:{}".format(dist))
        print("dist3D:{}".format(dist3D))           
    return img
    
def calcP():
    P = np.zeros((4,4), np.float32)
    K = np.zeros((3,4), np.float32)
    K[:,:-1] = intrinsic
    
    if not distance == 0:
        zeros = np.zeros(4)
        zeros[-1] = 1
        Rt = np.hstack((R, t))
        Rt = np.vstack((Rt, zeros))
        P = K @ Rt

    return P

def project3D(point):
    
    cam     = np.ones(3) # x
    cam[:-1]=point
    P = calcP()
    P = np.delete(P, 2, 1) # delete 3rd column
    W = np.linalg.inv(P) @ cam
    W = W / W[2] 
    W[2] = 0
    # X = X/w, Y = Y/w, Z = 0, w not needed
    
    return W

def calculateExtrinsics(image, exp, count):
    global object_points

    global R
    global t
    global distance
    global goodResult
    found, corners = cv2.findChessboardCorners(
        image, (board_w, board_h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found corners, refine
    if found == True:
        corners = cv2.cornerSubPix(
            image, corners, (11, 11), (-1, -1), criteria)
        if (corners.shape[0]==48):
            goodResult = True
            # cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]])
            # ret, r, t, inliners = cv2.solvePnPRansac(object_points, corners, intrinsic, distCoeff,None, None, True, 500, )
            
            ret, rvec, t = cv2.solvePnP(object_points, corners, intrinsic,
                                    distCoeff, None, None, False, cv2.SOLVEPNP_ITERATIVE)
            
            
            R, j = cv2.Rodrigues(rvec)
            C = np.matmul(np.linalg.inv(-R), t)
            distance = np.linalg.norm(C)
            # distance = np.linalg.norm(t)
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.drawChessboardCorners(
                image, (board_w, board_h), corners, found)
    return image

while(capture.isOpened()):
    _, image = capture.read()    
    image = cv2.flip( image, 1)  # mirrors image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,  w = image.shape[:2]
    newcameraintrinsic, roi = cv2.getOptimalNewCameraMatrix(intrinsic,distCoeff,(w,h),1,(w,h))

    #undistort
    dst = cv2.undistort(image, intrinsic, distCoeff, None, newcameraintrinsic)

    # # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.setMouseCallback('Raw',mouse_callback, raw)
    cv2.setMouseCallback('Undistorted',mouse_callback, undistorted)

    
    # image = drawLine(image, raw, (33,255,33))
    # cv2.imshow('Raw', image)
    
    dst = drawLine(dst, undistorted, (255,33,255))
    dst = calculateExtrinsics(dst, 0, count)

    # project3D(object_points, dst)
    if (dst.size>640*480):
        h, w, c = dst.shape
    else:
         h, w = dst.shape
    cv2.putText(dst, "Distance:{} cm".format(distance), (w-200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 33, 255), 1, cv2.LINE_AA)
    cv2.imshow('Undistorted', dst)

    k = cv2.waitKey(60) & 0xFF
    if k==27:    # Esc key to stop
        break
    if k == 32:  # Space -> snapshot
        saving = not saving

    if saving and goodResult:
        count += 1
        goodResult = False
        filename = directory + '/extr-{}-{}.png'.format(exp, count)
        cv2.imwrite(filename, dst)
        fs_write = cv2.FileStorage(
            './exp-{}/Extrinsics-{}.xml'.format(exp, count), cv2.FILE_STORAGE_WRITE)
        fs_write.write('R', R)
        fs_write.write('t', t)
        fs_write.write('distance', distance)
        fs_write.release()
capture.release()
cv2.destroyAllWindows()
