import numpy as np
import cv2
import glob


# termination criteria
# cv2.TERM_CRITERIA_EPS->stop the algorithm iteration if specified accuracy, epsilon, is reached
# cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER -> any of above criterias
# 30 = maxIter
# 0.001 = epsilon
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

board_w = 8  # horizontal enclosed corners on chessboard
board_h = 6 # vertical enclosed corners on chessboard

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(board_w-1,board_h-1,0)
objp = np.zeros((board_h*board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
image_points  = []  # 2d points in image plane.


exp = input("Please enter experiment number: ")
images = glob.glob('./exp-{}/s*.png'.format(exp))

for fname in images:
    image = cv2.imread(fname)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # cv2.findChessboardCorners(image, patternSize[, corners[, flags]]) → retval, corners
    #   image – Input image.
    #   patternSize – Number of inner corners per a chessboard (points_per_row, points_per_column) 
    #   corners – Array of detected corners

    found, corners = cv2.findChessboardCorners(gray_image, (board_w, board_h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

    # If found corners, refine
    if found == True:
        
        object_points.append(objp)
    
        # cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria) → corners
        #   corners – Initial coordinates of the input corners and refined coordinates provided for output.
        #   winSize – Half of the side length of the search window. For example, if winSize=Size(5,5) , then a  5*2+1 \times 5*2+1 = 11 \times 11 search window is used.
        #   zeroZone – Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
        #   criteria – Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.
        #  https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html      
        corners = cv2.cornerSubPix(
            gray_image, corners, (11, 11), (-1, -1), criteria)

        image_points.append(corners)
        
        
        # Draw and display the corners         
        # cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound) → image
        #   image – Destination image. It must be an 8-bit color image.
        #   patternWasFound – Parameter indicating whether the complete board was found or not. The return value of findChessboardCorners() should be passed here.
        image = cv2.drawChessboardCorners(image, (board_w, board_h), corners, found)
        cv2.imshow('Snapshot', image)
        cv2.waitKey(200)
cv2.destroyAllWindows()



print("\n\n *** Calibrating the camera now...")
intrinsic = np.zeros((3,3), np.float32)
intrinsic [0,0] = 1
intrinsic [1,1] = 1
# cv2.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) → retval, cameraMatrix, distCoeffs, rvecs, tvecs
#   cameraMatrix – Output 3x3 floating-point camera matrix  A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} . If CV_CALIB_USE_INTRINSIC_GUESS and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be initialized before calling the function.
#   distCoeffs – Output vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_board_h],[s_1, s_2, s_3, s_4]]) of 4, 5, 8 or 12 elements.
#   rvecs – Output vector of rotation vectors (see Rodrigues() ) estimated for each pattern view. That is, each k-th rotation vector together with the corresponding k-th translation vector (see the next output parameter description) brings the calibration pattern from the model coordinate space (in which object points are specified) to the world coordinate space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. M -1).
#   tvecs – Output vector of translation vectors estimated for each pattern view.
ret, intrinsic, distCoeff, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_image.shape[::-1],cv2.CALIB_USE_INTRINSIC_GUESS,criteria)

print ("rvecs", rvecs)
print ("tvecs", tvecs)
# check reprojection error
total_error = 0
for i in range(0, len(object_points)):
    image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], intrinsic, distCoeff)
    error = cv2.norm(image_points[i],image_points2, cv2.NORM_L2)/len(image_points2)
    total_error += error

print ("mean error: {} pixels ".format(total_error/len(object_points)))


print("Storing Intrinsics and Distortions files...\n")
print(intrinsic)
fs_write = cv2.FileStorage('Intrinsics.xml',cv2.FILE_STORAGE_WRITE)
fs_write.write('Intrinsics', intrinsic)
fs_write.release()

fs_write = cv2.FileStorage('Distortion.xml', cv2.FILE_STORAGE_WRITE)
fs_write.write('DistCoeffs', distCoeff)
fs_write.release()

# np.save('Intrinsics', intrinsic)
# np.save('Distortion', distCoeff)

