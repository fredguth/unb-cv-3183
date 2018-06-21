import numpy as np
import cv2
import glob
import os
import imutils
import math


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        # print("Lowval: ", low_val)
        # print("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(
            thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

filenames = glob.glob('./dataset/*/*')
videos = glob.glob('./videos/*.mp4')

#create directory for Ground Truth
for video in videos:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4', '')
  directory = './GT/'+video
  if not os.path.exists(directory):
      print (directory)
      os.makedirs(directory)


for filename in filenames:

  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  image = simplest_cb(image, 1)

  blurred_image = cv2.GaussianBlur(image, (3,3), 0)

  gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
  
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  mask = 255 - thresh
  # # mask = thresh
  # foreground = cv2.bitwise_and(image, image, mask=mask)
  
  # black = np.where((foreground[:, :, 0] < 1) &
  #                 (foreground[:, :, 1] < 1) & 
  #                 (foreground[:, :, 2] < 1))
  # foreground[black]=255


  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  for contour in contours:
    area = cv2.contourArea(contour)
    
    # only contours large enough to contain object
    if area > 300000:
      #print(area)
      # cv2.drawContours(image, contour, -1, (0,255,0),3)
      rect = cv2.minAreaRect(contour)
      
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      # cv2.drawContours(foreground, [box], 0, (0, 0, 255), -1)
      
      scale = 1 # cropping margin, 1 == no margin
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
      # # mostly for debugging purposes
      # cv2.circle(img_box, center, 10, (0, 255, 0), -1)

      M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

      # !!!Important: which image to crop
      cropped = cv2.getRectSubPix(image, size, center)
      cropped = cv2.warpAffine(cropped, M, size)

      croppedW = W if not rotated else H
      croppedH = H if not rotated else W

      image = cv2.getRectSubPix(
          cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))


  filename = filename.replace('dataset', 'GT')
  print (filename)
  cv2.imwrite(filename, image)
