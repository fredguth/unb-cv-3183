import numpy as numpy
import cv2

obj = np.load('obj.npy')
img = np.load('img.npy')

a,b = obj.shape
h = np.zeros((a, b+1), np.float32)
h[:,:-1] = obj
ones = np.ones(a)
h[:,3] = ones
obj = h

a, b = img.shape
h = np.zeros((a, b+1), np.float32)
h[:, :-1] = img
ones = np.ones(a)
h[:, 2] = ones
img = h

