import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob



def loadImages():
  ori = []
  gt =[]
  color = cv2.COLOR_BGR2RGB
  imageFolder = './sfa/ORI/'
  truthFolder = './sfa/GT/'
  testing = (951, 1118)
  
  for j in testing:
    filename = "{}img ({}).jpg".format(imageFolder, j)
    ori.append(cv2.cvtColor(cv2.imread(filename), color))
    filename = "{}img ({}).jpg".format(truthFolder, j)
    gt.append(cv2.cvtColor(cv2.imread(filename), color))

  ori = np.asarray(ori)
  gt == np.asarray(gt)
  return ori, gt


ori, gt = loadImages()

X = ori
Z = ori
Y =[]
W =[]
U =[]
T = np.load('./results/lim-YCbCrMask-2.npy')
T = np.asarray(T)
Y.append(T[0])
Y.append(T[-1])
Y = np.asarray(Y)
T = np.load('./results/bayesMask-2.npy')
T = np.asarray(T)
W.append(T[0])
W.append(T[-1])
W = np.asarray(W)
T = np.load('./results/knnMask-2.npy')
np.asarray(T)
s = 168
h = 576
w = 768
T = T.reshape((s,h,w))
U.append(T[0])
U.append(T[-1])
U = np.asarray(U)


Y[Y>0]=1
X = np.asarray(ori)

Y = cv2.bitwise_and(X,X,mask=Y)
Z = gt
Z = np.asarray(Z)
T = np.zeros((Z.shape), dtype=np.uint8)

T[W>0]=1
X = np.asarray(ori)
W = cv2.bitwise_and(X,X, mask=T)
T = np.zeros((Z.shape), dtype=np.uint8)
T = T[:,:,:,0]
# U = U.reshape((T.shape))
print (U.shape, T.shape, X.shape)
T = np.zeros((Z.shape), dtype=np.uint8)
T[U>0]=1
X = np.asarray(ori)
U = cv2.bitwise_and(X, X, mask=T)
Z = np.asarray(gt)
# X = loadImages(I=ori)
X = np.asarray(ori)

for j in range(0, len(X)):
    x = X[j]
    y = Y[j]
    w = W[j]
    z = Z[j]
    u = U[j]
    index = j*5+1
    plt.subplot(len(X), 5, index)
    plt.title("ORI")
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(x)
    plt.subplot(len(X), 5, index+1)
    plt.title("GT")
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(z)
    plt.subplot(len(X), 5,index+2)
    plt.title("CbCr")
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(y)
    plt.subplot(len(X), 5,index+3)
    plt.title("Bayes")
    plt.xticks([])
    plt.yticks([])
   
    plt.imshow(w)
    plt.subplot(len(X), 5, index+4)
    plt.title("K-NN")
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(u)

plt.subplots_adjust(wspace=0, hspace=0, left=0.1,
                      right=0.9, top=0.9, bottom=0.5)
plt.show()

# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1))

