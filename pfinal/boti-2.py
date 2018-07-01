
# coding: utf-8

# # Boticario Product Classification

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
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(0)


# In[3]:


# Uncomment the below if you need to reset your precomputed activations
get_ipython().system('rm -rf {PATH}tmp')


# In[65]:


def loadImages(filenames):
  imgs = []
  for j in range(0, len(filenames)):
    fn = str(filenames[j])
    try:
        im = cv2.imread(str(fn))
        if im is None: raise OSError(f'File not recognized by opencv: {fn}')
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e
    im = cv2.resize(im, (sz,sz))
    imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255)
  return imgs


# In[66]:


filenames = glob.glob('./data/boti/*/*')


# In[67]:


X = np.asarray(loadImages(filenames))


# In[68]:


X.shape


# In[80]:


y = []
for filename in filenames: 
  category = filename.split('/')[3]
  category = category.split('-')[0]
  y.append(category)
y = np.asarray(y).astype(np.int)


# In[81]:


y, y.shape


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify= y_train, test_size=0.2)


# In[83]:


y_train.shape


# In[84]:


trn = (X_train, y_train)
val = (X_val, y_val)
test = (X_test, y_test)


# Set model parameters:

# In[85]:


PATH = ""
sz=224 #resnet restriction
arch = resnet50
bs = 16


# ## Data augmentation

# In[86]:


# aug_tfms = [RandomRotate(5),
#             RandomLighting(0.07, 0.07)]


# In[87]:


tfms = tfms_from_model(arch, 
                       sz=sz,
                       aug_tfms=None
                       )


# In[88]:


trn


# In[89]:


data = ImageClassifierData.from_arrays(path=PATH, bs=bs,trn=trn, val=val, tfms=tfms, test=test)


# In[90]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
#     ax.set_xticks(np.linspace(0, 224, 9))
#     ax.set_yticks(np.linspace(0, 224, 9))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    #top left: b[:2], bottom_right: b[-2:]
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
        verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)


# ### Train dataset

# In[98]:


data.classes


# In[92]:


x, y = next(iter(data.trn_dl))
x = to_np(x)
y = to_np(y)
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):

#     image=data.trn_ds.denorm(x)[i]
    label= data.classes[y[i]]
    ax = show_img(image, ax=ax)
    draw_text(ax, (0,0), label)
plt.tight_layout()


# In[11]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# In[12]:


learn.lr_find(start_lr=1e-3, end_lr=1e2)


# In[13]:


learn.sched.plot(n_skip=0, n_skip_end=0)


# In[14]:


learn.fit(0.05, 8)


# In[15]:


x,y = next(iter(data.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x,preds = to_np(x),to_np(probs)
y = to_np(y)
print (accuracy_np(preds,y))
preds = np.argmax(preds, -1)


# In[16]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=data.val_ds.denorm(x)[i]
    b = data.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()


# In[17]:


x,y = next(iter(data.test_dl))


# In[18]:


dummy = predict_batch(learn.model, x)
dummy


# In[19]:


probs = F.softmax(dummy, -1)
probs


# In[20]:


x,preds = to_np(x),to_np(probs)


# In[21]:


y = to_np(y)


# In[22]:


accuracy_np(preds,y)


# In[23]:


preds = np.argmax(preds, -1)


# In[24]:


(preds==y)


# In[25]:


fig, axes = plt.subplots(4, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=data.test_ds.denorm(x)[i]
    b = data.classes[preds[i]]
    c = data.classes[y[i]]
    if b!=c:
        b = b + '/'+c
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
    
plt.tight_layout()


# In[26]:


targs = data.test_ds.y
targs = to_np(targs)


# In[27]:


targs.shape


# In[28]:


log_preds,_ = learn.TTA(n_aug=1, is_test=True)


# In[29]:


preds = np.mean(np.exp(log_preds),0)
preds = np.argmax(preds,axis=1)
preds.shape


# ### Resultado Pífio

# In[30]:


(preds==targs).mean()


# In[31]:


e = np.where(preds!=targs)
e = np.asarray(e)[0]
e, e.shape


# In[32]:


xe = []
for i in range(len(e)):
    xe.append(data.test_ds.get_x(e[i]))
xe = np.asarray(xe)
xe.shape


# In[33]:


# def loadImages(filenames):
#   imgs = []
#   for j in range(0, len(filenames)):
#     fn = str(PATH+filenames[j])
#     try:
#         im = cv2.imread(str(fn))
#         if im is None: raise OSError(f'File not recognized by opencv: {fn}')
#     except Exception as e:
#         raise OSError('Error handling image at: {}'.format(fn)) from e
#     im = cv2.resize(im, (sz,sz))
#     imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255)
#   return imgs

def get_sample(data, cat):
    path = data.trn_ds.path+"train/"+cat
    filename = str(os.listdir(path)[1])
    image = cv2.imread(path +"/"+ filename)
#     image = cv2.resize(image, (224,224))
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)/255
    
    return image


# In[48]:


preds.shape
guess = np.argmax(preds, 1)
guess.shape


# In[49]:


fig, axes = plt.subplots(5, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    if (i%2)==0:
        j = int (i/2)
        ima = xe[j]
        b = data.classes[guess[e[j]]]
        ax = show_img(ima, ax=ax)
        draw_text(ax, (0,0), b)
          
    else:
        j = int (i/2)
        imb = get_sample(data, data.classes[targs[e[j]]])
        b = data.classes[targs[e[j]]]
        ax = show_img(imb, ax=ax)
        draw_text(ax, (0,0), b)
plt.tight_layout()


# In[35]:


targs.shape


# In[36]:


xx = []
for i in range(len(targs)):
    xx.append(data.test_ds.get_x(i))
xx = np.asarray(xx)
xx.shape


# In[37]:


probs = F.softmax(VV(learn.predict(is_test=True)), -1)
probs


# In[38]:


x,preds = to_np(x),to_np(probs)


# In[39]:


accuracy_np(preds,targs)


# In[42]:


data.test_ds.fnames


# In[43]:


data.val_ds.fnames


# ## Validation <-> Test

# In[50]:


data = ImageClassifierData.from_paths(path=PATH, tfms=tfms, bs=bs, val_name='test', test_name='valid', test_with_labels=True)


# In[51]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# In[ ]:


learn.fit(0.05, 8)

