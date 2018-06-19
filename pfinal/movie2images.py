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
  directory = './data/boti/'+video
  if not os.path.exists(directory):
      os.makedirs(directory)

#extract images from videos and save in corresponding directory
for video in videos:
  filename = './videos/'+video+'.mp4'
  vidcap = cv2.VideoCapture(filename)
  
  success, image = vidcap.read()
  
  count = 0
  imageNumber =0
  success = True
  while success:
    if (count % 5)==0:
      image = imutils.rotate_bound(image, 90)
      imgname = './data/boti/'+video+'/'+video+'-'+str(imageNumber).zfill(5) + '.jpg'
      imageNumber +=1
      print (imgname)
      cv2.imwrite(imgname, image)     # save frame as JPEG file      
    
    success, image = vidcap.read()
    count += 1
  
  





# images = glob.glob('./exp-{}/s*.png'.format(exp))
# fs_read = cv2.FileStorage('./exp-0/Intrinsics.xml', cv2.FILE_STORAGE_READ)
# intrinsic = fs_read.getNode('Intrinsics').mat()
# fs_read.release()
# fs_read = cv2.FileStorage('./exp-0/Distortion.xml', cv2.FILE_STORAGE_READ)
# distCoeff = fs_read.getNode('DistCoeffs').mat()
# fs_read.release()
# count = 0

#     if k == ord("s"):
#         count +=1
#         filename = './exp-0/projection-{}.png'.format(count)
#         cv2.imwrite(filename, dst)

#     if saving and goodResult:
#         count += 1
#         goodResult = False
#         filename = directory + '/extr-{}-{}.png'.format(exp, count)
#         cv2.imwrite(filename, dst)
#         fs_write = cv2.FileStorage(
#             './exp-{}/Extrinsics-{}.xml'.format(exp, count), cv2.FILE_STORAGE_WRITE)
#         fs_write.write('R', R)
#         fs_write.write('t', t)
#         fs_write.write('distance', distance)
#         fs_write.release()
