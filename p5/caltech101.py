from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import numpy as np
import time
import cv2
from sklearn.model_selection import StratifiedKFold


# Set model parameters:

PATH = "data/caltech/direct/"
SZ=224 #resnet restriction
ARCH = resnet50

TFMS = tfms_from_model(arch, 
                       sz, 
                       aug_tfms=transforms_side_on, 
                       max_zoom=1.2)

def loadImages(filenames):
  imgs = []
  
  for j in range(0, len(filenames)):
    fn = str(PATH+filenames[j])
    try:
        im = cv2.imread(str(fn))
        if im is None: raise OSError(f'File not recognized by opencv: {fn}')
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e
    im = cv2.resize(im, (SZ,SZ))
    imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255)
  return imgs

def getXY():
    data = ImageClassifierData.from_paths(path=PATH, tfms=TFMS)

    train_files = data.trn_ds.fnames
    train_categories = data.trn_y
    train = data.trn_ds
    X = np.asarray(loadImages(train_files))
    Y = np.asarray(train_categories)
    return X, Y

def getFolds(X, Y):
  dummy = ImageClassifierData.from_paths(path=PATH, tfms=TFMS, trn_name='test')
  tst = np.asarray(loadImages(dummy.trn_ds.fnames))
  tst_ctg = dummy.trn_y
  skf = StratifiedKFold(n_splits=5)
  i =0
  data = []
  index = []
  for train_index, val_index in skf.split(X, Y):
      index.append((train_index, val_index))
      X_train = X[train_index]
      Y_train = Y[train_index]
      X_valid = X[val_index]
      Y_valid = Y[val_index]
      trn = (X_train, Y_train)
      val = (X_valid, Y_valid)
      data.append(ImageClassifierData.from_arrays(path=PATH, bs=4,
                                                  trn=trn, val=val, tfms=TFMS, test=(tst, tst_ctg)))
      i+=1
  return data, index      

def trainModels(data):
  for i in range(len(data)):
    m.append(ConvLearner.pretrained(arch, data[i], precompute=False))

def runSingleRate(models):
  print ('running single rate')
  start = time.time()
  for i in range(len(data)):
      startmodel = time.time()
      learn = models[i]
      learn.fit(0.012, 30)
      endmodel = time.time()
      format ('model{} took {}s'.format(i, (startmodel-endmodel)))
  end = time.time()
  duration = start - end
  print('single rate tool {}s in total'.format(duration))

def runCyclicalRate(models):
  print ('running cyclical rate')
  start = time.time()
  for i in range(len(data)):
      startmodel = time.time()
      learn = models[i]
      learn.fit(0.012, 5, cycle_len=1, cycle_mult=2)
      endmodel = time.time()
      format ('model{} took {}s'.format(i, (startmodel-endmodel)))
  end = time.time()
  duration = start - end
  print('Cyclical rate tool {}s in total'.format(duration))

def predict(models):
  preds=[]
  for i in range(len(data)):
    learn=models[i]
    log_preds, y=learn.TTA(n_aug=12, is_test=True)
    p=np.mean(np.exp(log_preds), 0)
    preds.append(p)
  mean=np.mean(preds, 0)
  return mean, y

X, Y = getXY()
data, indexes = getFolds(X, Y)
 
for j in range(5):
  m1 = trainModels(data)
  runSingleRate(m1)
  preds, y=predict(m1)
  result=accuracy_np(preds, y)
  print('result single rate {}: {}'.format(j, result))
np.save('m1', m1)

      

for j in range(5):
  m2 = trainModels(data)
  runCyclicalRate(m1)
  preds, y=predict(m1)
  result=accuracy_np(preds, y)
  print('result cyclical rate {}: {}'.format(j, result))
np.save('m2', m1)



