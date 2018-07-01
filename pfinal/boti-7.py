
# coding: utf-8

# # Boticario Product Classification

# In[3]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Using fastai lib

# In[4]:


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


# In[5]:


# Uncomment the below if you need to reset your precomputed activations
get_ipython().system('rm -rf {PATH}tmp')


# Set model parameters:

# In[6]:


PATH = "data/boti/"
sz=224 #resnet restriction
arch = resnet50
bs = 64


# ## Data Augmentation

# In[7]:


aug_tfms = [
   RandomDihedral(),
   RandomRotate(45, p=0.75, mode=cv2.BORDER_CONSTANT, tfm_y=TfmType.NO),
   RandomLighting(b=0, c=.5, tfm_y=TfmType.NO),
   RandomZoom(zoom_max=1.2),
   Cutout(n_holes=15, length=30, tfm_y=TfmType.NO)
#     AddPadding(pad=20, mode=cv2.BORDER_WRAP)
]


# In[8]:


tfms = tfms_from_model(arch, 
                       sz=sz,
                       aug_tfms=aug_tfms,
                       crop_type=CropType.NO
                       )


# In[10]:


data = ImageClassifierData.from_paths(path=PATH, bs=bs,tfms=tfms, test_name='mags_test', test_with_labels=True)


# ### Check data

# In[11]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
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


# In[12]:


def sample(dataloader):
    x, y = next(iter(dataloader))
    x = to_np(x)
    y = to_np(y)
    fig, axes = plt.subplots(3, 4, figsize=(12, 15))
    for i,ax in enumerate(axes.flat):
        image=data.trn_ds.denorm(x)[i]
        label= data.classes[y[i]]
        ax.grid(False)
        ax.set_title(label)
        ax = show_img(image, ax=ax)
        
plt.axis('off')
plt.tight_layout()
    


# ## Model

# Get a pretrained Resnet-50

# In[13]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# Find optimal learning rate

# In[24]:


lr = 0.02
lrs = np.array([lr/100, lr/10, lr])
learn.freeze_to(-2)
learn.fit(lrs/10, 5, cycle_len=1, cycle_mult=2)


# In[25]:


learn.sched.plot_lr()


# In[27]:


a = []
for i in range(5):
    log_preds,_ = learn.TTA(n_aug=7, is_test=True)
    probs = np.mean(np.exp(log_preds),0)
    targs = data.test_ds.y
    targs = to_np(targs)
    a.append(accuracy_np(probs, targs))


# In[28]:


a = np.asarray(a)
(a.mean(), a.std())


# In[17]:


probs_sorted = np.argsort(probs, axis=1)
probs_sorted.shape, targs.shape
preds = np.argmax(probs, 1)


# In[18]:


preds1=probs_sorted[:,-1]
preds2=probs_sorted[:,-2]
preds3=probs_sorted[:,-3]
preds4=probs_sorted[:,-4]
preds5=probs_sorted[:,-5]


# In[19]:


(preds1==targs).mean(), (preds2==targs).mean()


# In[20]:


((preds1==targs) | (preds2==targs) ).mean()


# In[21]:


e = np.asarray(np.where(preds!=targs))
e, len(e[0])


# In[22]:


import glob
def getSamples(category, dataset):
  imgs = []
  search = PATH+dataset+'/'+category+'/*'
  filenames = glob.glob(search)
  fi = min(len(filenames), 8)
  for j in range(0, fi):
    fn = str(filenames[j])
    try:
        im = cv2.imread(str(fn))
        if im is None: raise OSError(f'File not recognized by opencv: {fn}')
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e
    im = cv2.resize(im, (sz,sz))
    imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255)
  return imgs


# In[23]:



fig, axes = plt.subplots(9, 3, figsize=(9, 12))

for i,ax in enumerate(axes.flat):
    predcat = data.classes[preds[e[0,i]]]
    targcat = data.classes[targs[e[0,i]]]
    impred = getSamples(predcat, 'train')[0]
    imtarg = getSamples(targcat, 'mags_test')[0]
    im = np.hstack([impred, imtarg])
    ax = show_img(im, ax=ax)
    ax.set_title(predcat+'/'+targcat )
#     draw_text(ax, (0,0), txt=predcat+'/'+targcat)

plt.tight_layout()
    

