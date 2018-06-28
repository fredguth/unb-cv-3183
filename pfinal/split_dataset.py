import numpy as np
import cv2
import glob
import os
import imutils
from sklearn.model_selection import train_test_split
from shutil import copy2


filenames = glob.glob('./dataset/*/*')

y = []
for filename in filenames: 
  category = filename.split('/')[3]
  category = category[:5]
  y.append(category)

y = np.asarray(y)
print (y.shape)
X = filenames

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify= y_train, test_size=0.2)

#create directories

def copyFiles(dataset, filenames, categories):
  for i in range(len(filenames)):
    name = filenames[i].split('/')[3]
    directory = './data/'+dataset+'/'+categories[i]
    if not os.path.exists(directory):
        os.makedirs(directory)
    dst = directory +'/'+name
    print (dst)
    copy2(filenames[i], dst)

copyFiles('boti/train', X_train, y_train)
copyFiles('boti/valid', X_val, y_val)
copyFiles('boti/test', X_test, y_tes