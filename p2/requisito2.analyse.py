import numpy as np
import cv2

print("Experiments? (use space between experiment numbers [ex. 1 2 7]):")
experiments = [int(x) for x in input().split()]

intrinsics = []
distortions = []
for exp in experiments:
  fs_read = cv2.FileStorage(
      './exp-{}/Intrinsics.xml'.format(exp), cv2.FILE_STORAGE_READ)
  intrinsics.append(fs_read.getNode('Intrinsics').mat())
  fs_read.release()
  fs_read = cv2.FileStorage(
      './exp-{}/Distortion.xml'.format(exp), cv2.FILE_STORAGE_READ)
  distortions.append(fs_read.getNode('DistCoeffs').mat())
  fs_read.release()

intrinsics = np.asarray(intrinsics)
distortions = np.asarray(distortions)

mean_intrinsics = intrinsics.mean(axis=0)
mean_distortions = distortions.mean(axis=0)
std_intrinsics = intrinsics.std(axis=0)
std_distortions = distortions.std(axis=0)
fs_write = cv2.FileStorage(
    './exp-0/Intrinsics.xml'.format(exp), cv2.FILE_STORAGE_WRITE)
fs_write.write('Intrinsics', mean_intrinsics)
fs_write.release()

fs_write = cv2.FileStorage(
    './exp-0/Distortion.xml'.format(exp), cv2.FILE_STORAGE_WRITE)
fs_write.write('DistCoeffs', mean_distortions)
fs_write.release()

fs_write = cv2.FileStorage(
    './exp-0/Data.xml'.format(exp), cv2.FILE_STORAGE_WRITE)
fs_write.write('MeanIntrinsics', mean_intrinsics)
fs_write.write('StdIntrinsics', std_intrinsics)
fs_write.write('MeanDistortions', mean_distortions)
fs_write.write('StdDistortions', std_distortions)
fs_write.release()


