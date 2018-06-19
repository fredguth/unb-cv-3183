import numpy as np
import cv2
import glob
import os
import imutils
import json

filenames = glob.glob('./dataset/*/*')
data = {}
data['annotations'] = []
data['categories'] = [{"id": "background", "name": "background"}]
data['images'] = []

videos = glob.glob('./videos/*.mp4')

for video in videos:
  
  video = video.replace('./videos/', '')
  video = video.replace('.mp4', '')
  category_data = {}
  category_data['id'] = video
  category_data['name'] = video
  data['categories'].append(category_data)


# json_data = json.dumps(data)
# print (json_data)
for filename in filenames:
  print (filename)
  imageName = filename.replace('./dataset/', '')
  category, name = imageName.split('/')
  
  name = name.replace('.jpg', '')
  
  image = cv2.imread(filename, cv2.IMREAD_COLOR)

  h, w, c = image.shape
  image_data = {}
  image_data["id"] = name
  image_data["filename"] = filename
  image_data["width"] = w
  image_data["height"] = h
  data["images"].append(image_data)

  blurred_image = cv2.GaussianBlur(image, (3,3), 0)
  gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)  
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  mask = 255 - thresh

  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  annotation = {}
  for contour in contours:
    area = cv2.contourArea(contour)
    
    # only contours large enough to contain object
    if area > 100000:
      annotation['area'] = area
      rect = cv2.boundingRect(contour)
      x,y,w,h = rect
      annotation['bbox'] = [y, x, y+h, x+w]
      annotation['image_id'] = name
      annotation['category_id'] = category
      data['annotations'].append(annotation)

      
with open('data.json', 'w') as outfile:
    json.dump(data, outfile)

  # filename = filename.replace('dataset', 'GT')
  # print (filename)
  # cv2.imwrite(filename, image)
