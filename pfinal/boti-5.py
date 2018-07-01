
# coding: utf-8

# # Boticario Product Classification

# In[36]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Using fastai lib

# In[37]:


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


# In[38]:


# Uncomment the below if you need to reset your precomputed activations
get_ipython().system('rm -rf {PATH}tmp')


# Set model parameters:

# In[39]:


PATH = "data/boti/"
sz=224 #resnet restriction
arch = resnet50
bs = 64


# ## Data Augmentation

# In[40]:


aug_tfms = [
#     RandomDihedral()
   RandomRotate(90, p=0.75, mode=cv2.BORDER_CONSTANT, tfm_y=TfmType.NO),
#     RandomLighting(b=0.05, c=1, tfm_y=TfmType.NO),
#     RandomZoom(zoom_max=0.5),
   Cutout(n_holes=10, length=50, tfm_y=TfmType.NO)
#     AddPadding(pad=20, mode=cv2.BORDER_WRAP)
]


# In[41]:


tfms = tfms_from_model(arch, 
                       sz=sz,
                       aug_tfms=aug_tfms,
                       crop_type=CropType.NO
                       )


# In[79]:


data = ImageClassifierData.from_paths(path=PATH, bs=bs,tfms=tfms, test_name='test', test_with_labels=True)


# ### Check data

# In[84]:


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


# In[11]:


def sample(dataloader):
    x, y = next(iter(dataloader))
    x = to_np(x)
    y = to_np(y)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i,ax in enumerate(axes.flat):

        image=data.trn_ds.denorm(x)[i]
        label= data.classes[y[i]]
        ax = show_img(image, ax=ax)
        draw_text(ax, (0,0), label)
    plt.tight_layout()


# #### Training data

# In[80]:


sample(data.trn_dl)


# #### Validation data

# In[81]:


sample(data.val_dl)


# #### Test data

# In[82]:


sample(data.test_dl)


# ## Model

# Get a pretrained Resnet-50

# In[83]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# Find optimal learning rate

# In[84]:


learn.lr_find(start_lr=1e-3, end_lr=1e2)


# In[85]:


learn.sched.plot(n_skip=0, n_skip_end=0)


# In[86]:


learn.lr_find(start_lr=1e-1, end_lr=1e0)
learn.sched.plot(n_skip=0, n_skip_end=0)


# Train

# In[87]:


learn.fit(0.02, 8)


# ## Improve

# In[88]:


learn.fit(0.02, 5, cycle_len=1, cycle_mult=2)


# In[89]:


learn.sched.plot_lr()


# In[90]:


log_preds,_ = learn.TTA(n_aug=1, is_test=True)
probs = np.mean(np.exp(log_preds),0)
targs = data.test_ds.y
targs = to_np(targs)
accuracy_np(probs, targs)


# ## Mudando o domínio das Imagens Teste

# In[42]:


data = ImageClassifierData.from_paths(path=PATH, bs=bs,tfms=tfms, test_name='mags_test', test_with_labels=True)


# In[43]:


sample(data.test_dl)


# In[44]:


len(data.classes)
for label in data.classes:
    directory = '{}mags_test/{}'.format(PATH, label)
    if not os.path.exists(directory):
          os.makedirs(directory)


# ### Treinando Modelo

# In[45]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# In[46]:


learn.lr_find(start_lr=1e-3, end_lr=1e2)
learn.sched.plot(n_skip=0, n_skip_end=0)


# In[47]:


learn.fit(0.02, 12)


# In[48]:


learn.fit(0.02, 5, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()


# In[66]:


log_preds,_ = learn.TTA(n_aug=7, is_test=True)
probs = np.mean(np.exp(log_preds),0)
targs = data.test_ds.y
targs = to_np(targs)
accuracy_np(probs, targs)


# In[49]:


log_preds,_ = learn.TTA(n_aug=120, is_test=True)
probs = np.mean(np.exp(log_preds),0)
targs = data.test_ds.y
targs = to_np(targs)
accuracy_np(probs, targs)


# In[92]:


for i in range(5):
    log_preds,_ = learn.TTA(n_aug=7, is_test=True)
    probs = np.mean(np.exp(log_preds),0)
    targs = data.test_ds.y
    targs = to_np(targs)
    print (accuracy_np(probs, targs))


# In[91]:


sample(data.test_dl)


# In[52]:


probs, probs.shape


# In[53]:


probs_sorted = np.argsort(probs, axis=1)
probs_sorted.shape, targs.shape
preds = np.argmax(probs, 1)
preds


# In[54]:


preds1=probs_sorted[:,-1]
preds2=probs_sorted[:,-2]
preds3=probs_sorted[:,-3]
preds4=probs_sorted[:,-4]
preds5=probs_sorted[:,-5]


# In[55]:


(preds1==targs).mean(), (preds2==targs).mean()


# In[56]:


((preds1==targs) | (preds2==targs) |(preds3==targs)).mean()


# In[57]:


e = np.asarray(np.where(preds!=targs))
e, len(e[0])


# In[58]:


e[0,5], len(e[0])


# In[59]:


probs[e[0]].mean()


# In[60]:


a = np.setdiff1d(np.array(range(probs.shape[0])), e)


# In[61]:


(np.asarray(probs[a])).mean()


# In[62]:


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


# In[78]:


i = 32
predcat = data.classes[preds[e[0,i]]]
targcat = data.classes[targs[e[0,i]]]
impred = getSamples(predcat, 'train')[0]
imtarg = getSamples(targcat, 'mags_test')[0]
im = np.hstack([impred, imtarg])
ax = show_img(im, ax=ax)
plt.tight_layout()


# In[70]:


data.classes[preds[e[0,0]]], data.classes[targs[e[0,i]]]


# In[89]:



fig, axes = plt.subplots(8, 4, figsize=(9, 12))

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
    

