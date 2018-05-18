import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob



def loadImages(I, color=cv2.COLOR_BGR2RGB):
  Iout = []
  for j in range(0, len(I)):
    Iout.append(cv2.cvtColor(cv2.imread(I[j]), color))
  return Iout


ori = glob.glob('./dataset/test/ORI/*.jpg')
gt = glob.glob('./dataset/test/GT/*.jpg')

X = loadImages(I=ori)
X = np.asarray(X)
Z = X
Y = np.load('./results/lim-YCbCrMask.npy')
W = np.load('./results/bayesMask-1.npy')
U = np.load('./results/knnMask-1.npy')

W = np.asarray(W)
Y = np.asarray(Y)
U = np.asarray(U)
Y[Y>0]=1
T = Z # transforma shape
T[W<1]=1


Y = cv2.bitwise_and(Z,Z,mask=Y)
W = cv2.bitwise_and(Z,Z, mask=T)
T = Z[:,:,:,0]
U = U.reshape((T.shape))
T = Z
T[U<1]=1
U = cv2.bitwise_and(Z, Z, mask=T)
Z = loadImages(I=gt)
Z = np.asarray(Z)
X = loadImages(I=ori)
X = np.asarray(X)
for j in range(0, len(X)):
    x = X[j]
    y = Y[j]
    w = W[j]
    z = Z[j]
    u = U[j]
    index = j*5+1
    plt.subplot(len(X), 5, index)
    plt.xticks([])
    plt.yticks([])
    plt.title('ORI')
    plt.imshow(x)
    plt.subplot(len(X), 5, index+1)
    plt.xticks([])
    plt.yticks([])
    plt.title('GT')
    plt.imshow(z)
    plt.subplot(len(X), 5,index+2)
    plt.xticks([])
    plt.yticks([])
    plt.title('CbCr')
    plt.imshow(y)
    plt.subplot(len(X), 5,index+3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Bayes')
    plt.imshow(w)
    plt.subplot(len(X), 5, index+4)
    plt.xticks([])
    plt.yticks([])
    plt.title('K-nn')
    plt.imshow(u)

plt.subplots_adjust(wspace=0, hspace=0, left=0.1,
                      right=0.9, top=0.9, bottom=0.5)
plt.show()

# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1))

