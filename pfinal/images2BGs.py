import numpy as np
import cv2
import glob
import os
import imutils

filenames = glob.glob('./dataset/*/*')
videos = glob.glob('./videos/*.mp4')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 400,400)
#create directory for Ground Truth
for video in videos:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4', '')
  directory = './BG/'+video
  if not os.path.exists(directory):
      print (directory)
      os.makedirs(directory)


for filename in filenames:

  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  blurred_image = cv2.GaussianBlur(image, (3,3), 0)

  gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
  
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  mask = 255 - thresh
  


  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  for contour in contours:
    area = cv2.contourArea(contour)
    
    # only contours large enough to contain object
    if area > 100000:
      
      # cv2.drawContours(image, contour, -1, (0,255,0),3)
      rect = cv2.minAreaRect(contour)
      
      box = cv2.boxPoints(rect)
      box = np.int0(box)

      cv2.drawContours(image, [box], 0, (0, 255, 0), -1)
      
      
  filename = filename.replace('dataset', 'BG')
  print (filename)
  cv2.imwrite(filename, image)
  
    
