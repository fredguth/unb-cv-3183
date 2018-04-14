import numpy as np
import cv2
import os

exp = input("Please enter experiment number: ")
cwd = os.getcwd()
directory = cwd + '/exp-'+exp
if not os.path.exists(directory):
    os.makedirs(directory)

capture = cv2.VideoCapture(0)
# capture.set(3, 640)
# capture.set(4, 360)
cv2.namedWindow("Snapshot")
cv2.namedWindow("Raw")
snapshot = None
count = 0
while True:
  ret, image = capture.read()
  image = cv2.flip( image, 1)  # mirrors image
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("Raw", image)
  if count:
    cv2.imshow("Snapshot", snapshot)
    cv2.moveWindow("Snapshot", 640,43)    
    
  # Snapshot ('Space') or Stop ('ESC')  
  k = cv2.waitKey(15)
  if k == 27:  # Esc -> Stop
    break
  if k == 32:  # Space -> snapshot
    snapshot = gray_image
    count +=1
    filename = directory + '/snap-{}-{}.png'.format(exp, count)
    print (filename)
    cv2.imwrite(filename, gray_image)
 

cv2.destroyAllWindows()