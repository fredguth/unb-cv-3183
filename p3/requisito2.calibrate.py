# Based on https://gist.github.com/jensenb
import cv2
import numpy as np

def in_front_of_both_cameras(left_points, right_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    
    for left, right in zip(left_points, right_points):
        left_z = np.dot(rot[0, :] - right[0]*rot[2, :], trans) / np.dot(rot[0, :] - right[0]*rot[2, :], right)
        left_3d_point = np.array([left[0] * left_z, right[0] * left_z, left_z])
        right_3d_point = np.dot(rot.T, left_3d_point) - np.dot(rot.T, trans)

        if left_3d_point[2] < 0 or right_3d_point[2] < 0:
            return False

    return True


# load stereo images
imgL = cv2.imread("./results/imgR.png")
imgR = cv2.imread("./results/imgL.png")

# get camera parameters
fs_read = cv2.FileStorage('./exp-0/Intrinsics.xml', cv2.FILE_STORAGE_READ)
K = fs_read.getNode('Intrinsics').mat()
fs_read.release()
fs_read = cv2.FileStorage('./exp-0/Distortion.xml', cv2.FILE_STORAGE_READ)
d = fs_read.getNode('DistCoeffs').mat()
fs_read.release()

K_inv = np.linalg.inv(K)

# images already undistorted when captured
imgL_undist = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_undist = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


# generate lists of correspondences
# image_size = imgL.shape
# find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

# left_found, left_corners = cv2.findChessboardCorners(imgL_undist, (8, 6), flags=find_chessboard_flags)
# right_found, right_corners = cv2.findChessboardCorners(imgR_undist, (8, 6), flags=find_chessboard_flags)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# if left_found and (len(left_corners)==48):
#     print('Left corners found')
#     cv2.cornerSubPix(imgL_undist, left_corners, (11, 11), (-1, -1), criteria)
#     tempL = imgL_undist
#     tempL = cv2.cvtColor(tempL, cv2.COLOR_GRAY2BGR)
#     tempL = cv2.drawChessboardCorners(tempL, (8, 6), left_corners, left_found)
#     # cv2.imshow('left', tempL)
# else:
#     print('Left corners not found')
# if right_found and (len(right_corners) == 48):
#     print('Right corners  found')
#     cv2.cornerSubPix(imgR_undist, right_corners, (11, 11), (-1, -1), criteria)
#     tempR = imgR_undist
#     tempR = cv2.cvtColor(tempR, cv2.COLOR_GRAY2BGR)
#     tempR = cv2.drawChessboardCorners(tempR, (8, 6), right_corners, right_found)
#     # cv2.imshow('right', tempR)
# else:
#     print ('Right corners not found')

# left_match_points = left_corners[:, 0, :]
# right_match_points = right_corners[:, 0, :]


detector = cv2.xfeatures2d.SURF_create(250)
left_key_points, left_descriptors = detector.detectAndCompute(
    imgL_undist, None)
right_key_points, right_descriptos = detector.detectAndCompute(
    imgR_undist, None)
# imgL_undist = cv2.drawKeypoints(imgL_undist, left_key_points, None)
# cv2.imshow('left', imgL_undist)
# imgR_undist = cv2.drawKeypoints(imgR_undist, right_key_points, None)
# cv2.imshow('right', imgR_undist)
# match descriptors
matcher = cv2.BFMatcher(cv2.NORM_L1, True)
matches = matcher.match(left_descriptors, right_descriptos)

# generate lists of point correspondences
left_match_points = np.zeros((len(matches), 2), dtype=np.float32)
right_match_points = np.zeros_like(left_match_points)


img = cv2.drawMatches(imgL_undist, left_key_points, imgR_undist, right_key_points, matches, imgL_undist)
cv2.imshow('matches', img)
for i in range(len(matches)):
    left_match_points[i] = left_key_points[matches[i].queryIdx].pt
    right_match_points[i] = right_key_points[matches[i].trainIdx].pt

# estimate fundamental matrix
F, mask = cv2.findFundamentalMat(
    left_match_points, right_match_points, cv2.FM_RANSAC, 0.1, 0.99)

# decompose into the essential matrix
E = K.T.dot(F).dot(K)

# decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
U, S, Vt = np.linalg.svd(E)
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)


# iterate over all correspondences used in the estimation of the fundamental matrix
# left_matches = left_corners[:, 0, :]
# right_matches = right_corners[:, 0, :]
left_inliers = []
right_inliers = []
for i in range(len(mask)):
    if mask[i]:
        # normalize and homogenize the image coordinates
        left_inliers.append(
            K_inv.dot([left_match_points[i][0], left_match_points[i][1], 1.0]))
        right_inliers.append(
            K_inv.dot([right_match_points[i][0], right_match_points[i][1], 1.0]))


# Determine the correct choice of second camera matrix
# only in one of the four configurations will all the points be in front of both cameras
# First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
R = U.dot(W).dot(Vt)
T = U[:, 2]
print ('1')
if not in_front_of_both_cameras(left_inliers, right_inliers, R, T):
    print ('2')
    # Second choice: R = U * W * Vt, T = -u_3
    T = - U[:, 2]
    if not in_front_of_both_cameras(left_inliers, right_inliers, R, T):
        print('3')
        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not in_front_of_both_cameras(left_inliers, right_inliers, R, T):
            print('4')
            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

#perform the rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K, d, K, d, imgL.shape[:2], R, T, alpha=1.0)
mapx1, mapy1 = cv2.initUndistortRectifyMap(
    K, d, R1, K, imgL.shape[:2], cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(
    K, d, R2, K, imgR.shape[:2], cv2.CV_32F)
img_rect1 = cv2.remap(imgL, mapx1, mapy1, cv2.INTER_LINEAR)
img_rect2 = cv2.remap(imgR, mapx2, mapy2, cv2.INTER_LINEAR)
# cv2.imshow('img_rect1', img_rect1)
# cv2.imshow('img_rect2', img_rect1)
# draw the images side by side
total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
              img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# draw horizontal lines every 25 px accross the side by side image
for i in range(20, img.shape[0], 25):
    cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

# cv2.imshow('rectified', img)
while (True):
    k = cv2.waitKey(60) & 0xFF
    if k == 27:    # Esc key to stop
            cv2.destroyAllWindows()
