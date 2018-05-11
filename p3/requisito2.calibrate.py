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


# generate lists of correspondences
detector = cv2.xfeatures2d.SIFT_create()
left_key_points, left_descriptors = detector.detectAndCompute(
    imgL, None)
right_key_points, right_descriptos = detector.detectAndCompute(
    imgR, None)


# match descriptors
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(left_descriptors, right_descriptos, k=2)
# Need to draw only good matches, so create a mask
good_matches = []
# ratio test as per Lowe's paper
for m, n in matches:
    if m.distance < 0.55*n.distance:
        good_matches.append(m)

left_match_points = np.float32([left_key_points[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
right_match_points = np.float32(
    [right_key_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
print ('left_match_points', left_match_points.shape)
draw_params = dict( matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = None,
                    flags = 2)
img = cv2.drawMatches(imgL, left_key_points, imgR, right_key_points, good_matches, None, **draw_params)
cv2.imshow('matches', img)


# estimate fundamental matrix
F, mask = cv2.findFundamentalMat(
    left_match_points, right_match_points, cv2.FM_RANSAC, 0.1, 0.99)

# decompose into the essential matrix
E = K.T.dot(F).dot(K)

# decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
U, S, Vt = np.linalg.svd(E)
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)


# iterate over all correspondences used in the estimation of the fundamental matrix
left_inliers = []
right_inliers = []
left_match_points = left_match_points[:,0,:]
right_match_points = right_match_points[:, 0, :]
for i in range(len(mask)):
    if mask[i]:
        # normalize and homogenize the image coordinates
        left_inliers.append(
            K_inv.dot([left_match_points[i][0], left_match_points[i][1], 1.0]))
        right_inliers.append(
            K_inv.dot([right_match_points[i][0], right_match_points[i][1], 1.0]))


# #perform the rectification
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     K, d, K, d, imgL.shape, R, T, alpha=1.0)
# mapx1, mapy1 = cv2.initUndi# Determine the correct choice of second camera matrix
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
print ('R', R)
print ('T', T)
mapx1, mapy1 = cv2.initUndistortRectifyMap(
    K, d, R, K, imgL.shape, cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(
    K, d, R2, K, imgR.shape, cv2.CV_32F)
img_rect1 = cv2.remap(imgL, mapx1, mapy1, cv2.INTER_LINEAR)
img_rect2 = cv2.remap(imgR, mapx2, mapy2, cv2.INTER_LINEAR)
cv2.imshow('img_rect1', img_rect1)
cv2.imshow('img_rect2', img_rect2)
# # draw the images side by side
# total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
#               img_rect1.shape[1] + img_rect2.shape[1], 3)
# img = np.zeros(total_size, dtype=np.uint8)
# img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
# img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# # draw horizontal lines every 25 px accross the side by side image
# for i in range(20, img.shape[0], 25):
#     cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

# # cv2.imshow('rectified', img)
while (True):
    k = cv2.waitKey(60) & 0xFF
    if k == 27:    # Esc key to stop
            cv2.destroyAllWindows()
