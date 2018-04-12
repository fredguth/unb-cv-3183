import numpy as np
import cv2

# termination criteria
# cv2.TERM_CRITERIA_EPS->stop the algorithm iteration if specified accuracy, epsilon, is reached
# cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER -> any of above criterias
# 30 = maxIter
# 0.001 = epsilon
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

n_boards = int(input("Enter the number of spanspots: "))

# frame_step => # frames to be skipped
frame_step = int(input("Enter the number of frames to skip: "))

board_w = 8  # horizontal enclosed corners on chessboard
board_h = 6  # vertical enclosed corners on chessboard

board_total = board_w * board_h

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 360)
cv2.namedWindow("Snapshot")
cv2.namedWindow("Raw")

# 2d points in image plane.
image_points =  np.zeros((n_boards*board_total, 2), np.float32)

# 3d point in real world space
object_points = np.zeros((n_boards*board_total, 3), np.float32)

point_counts =  np.zeros((n_boards, 1), np.float32)

# Note:
# 	Intrinsic Matrix - 3x3			   Lens Distorstion Matrix - 4x1
# 		[fx 0 cx]							        [k1 k2 p1 p2   k3(optional)]
# 		[0 fy cy]
# 		[0  0  1]

intrinsic_matrix  = np.zeros((3,3), np.float32)
distortion_coeffs = np.zeros((4,1), np.float32)

successes = 0
step = 0
frame =  0

while (successes < n_boards):
  frame +=1
  if ((frame % frame_step) ==0):      # skip frames

    # Find the chess board corners
    # cv2.findChessboardCorners(image, patternSize[, corners[, flags]]) → retval, corners
    #   image – Input image.
    #   patternSize – Number of inner corners per a chessboard (points_per_row, points_per_column) 
    #   corners – Array of detected corners
    # ret, corners = cv2.findChessboardCorners(gray, (board_h, board_w), None)

    # If found, refine points
    # if ret == True:
    if True:
      ret, image = capture.read()
      cv2.imshow("Raw", image)
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria) → corners
      #   corners – Initial coordinates of the input corners and refined coordinates provided for output.
      #   winSize – Half of the side length of the search window. For example, if winSize=Size(5,5) , then a  5*2+1 \times 5*2+1 = 11 \times 11 search window is used.
      #   zeroZone – Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
      #   criteria – Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.
      #  https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html

      # refined_corners = cv2.cornerSubPix(gray, found, (11, 11), (-1, -1), criteria)
      
      # Draw and display the corners
      # cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound) → image
      #   image – Destination image. It must be an 8-bit color image.
      #   patternWasFound – Parameter indicating whether the complete board was found or not. The return value of findChessboardCorners() should be passed here.

      # image = cv2.drawChessboardCorners(image, (board_h, board_w), refined_corners, ret)
      cv2.imshow("Snapshot", gray_image)
      cv2.moveWindow('Snapshot', 640, 43)
      
      # Handle pause/unpause ('Space') and stop ('ESC')
      k = cv2.waitKey(15)
      if k == 27:  # Esc -> Stop
        break
      if k == 32:  # Space -> pause
        k=0
        while((not k==27) and (not k==32)):
          k = cv2.waitKey(250)
      
cv2.destroyAllWindows()