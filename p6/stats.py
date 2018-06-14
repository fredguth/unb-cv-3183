import numpy as np
import scipy.spatial as sp
import cv2
import time
import math

car1Basic = np.load("car1-stats.npy")
car1Kalman = np.load("car1-stats-kalman.npy")
car2Basic = np.load("car2-stats.npy")
car2Kalman = np.load("car2-stats-kalman.npy")

print (car1Basic.shape, car1Kalman.shape, car2Basic.shape, car2Kalman.shape)
print(car1Basic[200], car1Kalman[200], car2Basic[200], car2Kalman[200])

acc1b = car1Basic[:,0]
acc1b = acc1b[acc1b != np.array(None)]
rbs1b = car1Basic[:,1]
rbs1b = rbs1b[rbs1b!=np.array(None)]
acc1k = car1Kalman[:, 0]
acc1k = acc1k[acc1k != np.array(None)]
rbs1k = car1Kalman[:, 1]
rbs1k = rbs1k[rbs1k != np.array(None)]
acc2b = car2Basic[:, 0]
acc2b = acc2b[acc2b != np.array(None)]
rbs2b = car2Basic[:, 1]
rbs2b = rbs2b[rbs2b != np.array(None)]
acc2k = car2Kalman[:, 0]
acc2k = acc2k[acc2k != np.array(None)]
rbs2k = car2Kalman[:, 1]
rbs2k = rbs2k[rbs2k != np.array(None)]

acc1b = np.mean(acc1b)
rbs1b = np.mean(rbs1b)
acc1k = np.mean(acc1k)
rbs1k = np.mean(rbs1k)
acc2b = np.mean(acc2b)
rbs2b = np.mean(rbs2b)
acc2k = np.mean(acc2k)
rbs2k = np.mean(rbs2k)

print ((acc1b, rbs1b), (acc2b, rbs2b))
print ((acc1k, rbs1k), (acc2k, rbs2k))