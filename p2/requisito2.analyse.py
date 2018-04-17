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
print (intrinsics[0])
print (mean_intrinsics)
print (std_intrinsics)
print ('----------')
print(distortions[0])
print(mean_distortions)
print(std_distortions)

# print (intrinsics[:,0,0])
# print (np.mean(intrinsics[:,0,0]))
# print(intrinsics[:, 1, 0])
# print(np.mean(intrinsics[:, 1, 0]))
# print(intrinsics[:, 2, 0])
# print(np.mean(intrinsics[:, 1, 0]))
