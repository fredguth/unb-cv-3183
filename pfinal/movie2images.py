import glob
import numpy as np
import os
import cv2
import imutils



filenames = glob.glob('./videos/*.mp4')

# create array with video names and directories with video names
videos = []
directories = []
for video in filenames:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4','')
  videos.append(video)
  directory = './dataset/'+video
  if not os.path.exists(directory):
      os.makedirs(directory)

#extract images from videos and save in corresponding directory
for video in videos:
  filename = './videos/'+video+'.mp4'
  vidcap = cv2.VideoCapture(filename)
  
  success, image = vidcap.read()
  
  
  imageNumber =0
  success = True
  while success:
  
    image = imutils.rotate_bound(image, 90)
    imgname = './dataset/'+video+'/'+video+'-'+str(imageNumber).zfill(5) + '.jpg'
    imageNumber +=1
    print (imgname)
    cv2.imwrite(imgname, image)     # save frame as JPEG file      
  
    success, image = vidcap.read()
  
  
