
# coding: utf-8

# # CalTech 101

# In[1]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Using fastai lib

# In[2]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import time
from sklearn.model_selection import StratifiedKFold


# In[3]:


# Uncomment the below if you need to reset your precomputed activations
get_ipython().system('rm -rf {PATH}tmp')


# Set model parameters:

# In[4]:


PATH = "data/caltech/direct/"
SZ=224 #resnet restriction
ARCH = resnet50
arch = ARCH

TFMS = tfms_from_model(ARCH, 
                       SZ, 
                       aug_tfms=transforms_side_on, 
                       max_zoom=1.2)


# In[5]:


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


# In[6]:


def getXY():
    data = ImageClassifierData.from_paths(path=PATH, tfms=TFMS)

    train_files = data.trn_ds.fnames
    train_categories = data.trn_y
    train = data.trn_ds
    X = np.asarray(loadImages(train_files))
    Y = np.asarray(train_categories)
    return X, Y


# In[7]:


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


# In[8]:


def trainModels(data):
    m=[]  
    for i in range(len(data)):
        m.append(ConvLearner.pretrained(arch, data[i], precompute=False))
    return m


# In[9]:


def runSingleRate(models):
  print ('running single rate')
  start = time.time()
  for i in range(len(data)):
      startmodel = time.time()
      learn = models[i]
      learn.fit(0.012, 30)
      endmodel = time.time()
      duration = endmodel - startmodel
      print('model{} took {}s'.format(i, duration))
  finish = time.time()
  duration = finish - start
  print('single rate tool {}s in total'.format(duration))


# In[10]:


def runCyclicalRate(models):
  print ('running cyclical rate')
  start = time.time()
  for i in range(len(data)):
      startmodel = time.time()
      learn = models[i]
      learn.fit(0.012, 5, cycle_len=1, cycle_mult=2)
      endmodel = time.time()
      duration = endmodel - startmodel
      print ('model {} took {}s'.format(i, duration))
  end = time.time()
  duration = end - start
  print('Cyclical rate tool {}s in total'.format(duration))  


# In[11]:


def predictImages(models):
    print ('predicting images')
    preds=[]
    start = time.time()  
    for i in range(len(data)):
        s = time.time()
        learn=models[i]
        log_preds, y=learn.TTA(is_test=True)
        p=np.mean(np.exp(log_preds), 0)
        preds.append(p)
        e = time.time()
        d = e-s
        print ('predicting for model {} took {}s'.format(i,d))
    mean=np.mean(preds, 0)
    end =  time.time()
    duration = end - start
    print('Predicting took {}s'.format(duration))
    return mean, y


# In[12]:


X, Y = getXY()


# In[13]:


data, indexes = getFolds(X, Y)


# In[14]:


for j in range(3):
  m1 = trainModels(data)
  runSingleRate(m1)
  preds, y =predictImages(m1)
  result=accuracy_np(preds, y)
  print('result single rate {}: {}'.format(j, result))


# In[15]:


get_ipython().run_cell_magic('time', '', "for j in range(3):\n  m2 = trainModels(data)\n  runCyclicalRate(m2)\n  preds, y =predictImages(m2)\n  result=accuracy_np(preds, y)\n  print('result cyclical rate {}: {}'.format(j, result))")

