import numpy as np
import cv2
import glob
import os
import imutils

filenames = glob.glob('./dataset/*/*')
videos = glob.glob('./videos/*.mp4')

for video in videos:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4', '')
  directory = './mask/'+video
  if not os.path.exists(directory):
      print (directory)
      os.makedirs(directory)


# for filename in filenames[0]:
filename = filenames[0]
image = cv2.imread(filename)
blurred_image = cv2.GaussianBlur(image, (5,5), 0)

gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 9, .1, .1)
# edged = cv2.Canny(gray, 30, 200)
ret, thresh = cv2.threshold(gray,127, 255, cv2.THRESH_BINARY)
mask = 255 - thresh
# # mask = thresh
foreground = cv2.bitwise_and(image, image, mask=mask)
print (foreground.shape)
black = np.where((foreground[:, :, 0] < 1) &
                 (foreground[:, :, 1] < 1) & 
                 (foreground[:, :, 2] < 1))
foreground[black]=255
_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
  area = cv2.contourArea(contour)
  
  if area > 5000:
    
    # cv2.drawContours(image, contour, -1, (0,255,0),3)
    rect = cv2.minAreaRect(contour)
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
    
    scale = 1.1 # cropping margin, 1 == no margin
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(scale*(x2-x1)), int(scale*(y2-y1)))
    # # again this was mostly for debugging purposes
    # cv2.circle(img_box, center, 10, (0, 255, 0), -1)

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(foreground, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    image = cv2.getRectSubPix(
        cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))

# ctr = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
# cv2.drawContours(image, [ctr], 0, (0, 255, 0), -1)




filename = filename.replace('dataset', 'mask')
print (filename)
cv2.imwrite(filename, image)
