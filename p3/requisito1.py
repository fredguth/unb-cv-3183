import numpy as np
import cv2


imgL = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeL.png', cv2.IMREAD_GRAYSCALE))
imgR = cv2.pyrDown(cv2.imread('./imgs-estereo/aloeR.png', cv2.IMREAD_GRAYSCALE))

window_size = 9
min_disp = 0
max_disp = 112

def normalize(matrix):
  mat = matrix
  min_mat = mat.min()
  max_mat = mat.max()
  mat = mat - min_mat
  mat = mat/max_mat
  return mat

def project3D(dispMatrix):
  disp = dispMatrix
  l, c = disp.shape
  # print (h,2)
  world = np.zeros((l, c, 3), np.float32)
  b = 120
  f = 25
  for h in range(0,l):
    for w in range (0,c):
      xL = w
      xR = w+disp[h,w]
      yL = h
      yR = h 
      if not (xL-xR==0):     
        X = b*(xL+xR)/2*(xL-xR)
        Y = b*(yL+yR)/2*(xL-xR)
        Z = b*f/(xL-xR)
      world[h,w] = [X, Y, Z]
  return world
# stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
#                         numDisparities=max_disp,
#                         blockSize=window_size,
#                         P1=8*3*window_size**2,
#                         P2=32*3*window_size**2,
#                         disp12MaxDiff=1,
#                         uniquenessRatio=10,
#                         speckleWindowSize=100,
#                         speckleRange=32
#                         )
stereo = cv2.StereoBM_create(
                             numDisparities=max_disp
                             ,
                             blockSize=window_size
                             )
print ('computing disparity...')
disp = stereo.compute(imgL, imgR)
disparity_visual = np.zeros((disp.shape), dtype=np.uint8)
cv2.normalize(
    disp, disparity_visual, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
bsImg = np.array(disparity_visual)

print ('imageL shape', imgL.shape)
print('disp shape', disp.shape)
world = project3D(disp)
print ('world shape', world.shape)
depth = world[:,:,2]
depth = (depth-depth.min())
print('depth shape', depth.shape)
print ('depth min', depth.min())
print('depth max', depth.max())
depth = (depth/depth.max())


cv2.imshow('basic', bsImg)
cv2.imshow('depth', depth)

while(True):

  k = cv2.waitKey(60) & 0xFF
  if k == 27:    # Esc key to stop
        break

cv2.destroyAllWindows()

# print (disparity.shape)
# print(disparity)
# while(True):
#   cv2.imshow('Depth', disparity)
#   k = cv2.waitKey(60) & 0xFF
#   if k == 27:    # Esc key to stop
#     break

# w, h, c = left.shape
# k = 3   # kernel size 
# h_k = 1 # half k
# # #matrix w, h, (x', cost)
# disparity_matrix = np.full((w,h,2), math.inf, dtype = np.float32)
# max_disparity = 15
# for i in range(h_k, h-h_k):
#   for j in range(h_k, h-h_k):
#     template = left[j-h_k:j+h_k+1, i-h_k:i+h_k+1]
#     for d in range(0,max_disparity):
#       candidate = right[j+d-h_k: j+d+h_k+1, i-h_k: i+h_k+1]
#       print (template.shape)
#       print (candidate.shape)
#       candidate_cost = sp.distance_matrix(template.reshape(k*k,3), candidate.reshape(k*k,3), p=1)
#       print (candidate_cost)
#       # if (candidate_cost < disparity_matrix(j, i, [1])):
#       #   disparity_matrix[j, i, 0] = j+d
#       #   disparity_matrix[j, i, 1] = candidate_cost

# print (disparity_matrix.shape)
# print (disparity_matrix)

