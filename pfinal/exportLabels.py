import glob
import numpy as np
import os
import cv2



filenames = glob.glob('./videos/*.mp4')

# create array with video names and directories with video names
videos = []
for video in filenames:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4','')
  videos.append(video)
  
with open('classes.txt', 'w') as file:
    for video in videos:
      file.write(video+"\n")
file.close()