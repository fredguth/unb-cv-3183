
# coding: utf-8

# # Boticario Product Classification

# In[124]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Using fastai lib

# In[125]:


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


# In[126]:


# Uncomment the below if you need to reset your precomputed activations
get_ipython().system('rm -rf {PATH}tmp')


# Set model parameters:

# In[127]:


PATH = "data/boti/"
sz=224 #resnet restriction
arch = resnet50
bs = 16


# ## Data

# In[128]:


# aug_tfms = [RandomRotate(5),
#             RandomLighting(0.07, 0.07)]


# In[129]:


tfms = tfms_from_model(arch, 
                       sz=sz,
                       aug_tfms=None
                       )


# In[130]:


data = ImageClassifierData.from_paths(path=PATH, bs=bs,tfms=tfms, test_name='test', test_with_labels=True)


# ### Check data

# In[131]:


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


# In[132]:


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

# In[133]:


sample(data.trn_dl)


# #### Validation data

# In[134]:


sample(data.val_dl)


# #### Test data

# In[135]:


sample(data.test_dl)


# ## Model

# Get a pretrained Resnet-50

# In[136]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# Find optimal learning rate

# In[137]:


learn.lr_find(start_lr=1e-3, end_lr=1e2)


# In[138]:


learn.sched.plot(n_skip=0, n_skip_end=0)


# Train

# In[139]:


learn.fit(0.05, 8)


# ## Check results

# Run results for a validation batch

# In[140]:


x,y = next(iter(data.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x,preds = to_np(x),to_np(probs)
y = to_np(y)
print (accuracy_np(preds,y))
preds = np.argmax(preds, -1)


# In[141]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=data.val_ds.denorm(x)[i]
    b = data.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()


# In[142]:


x,y = next(iter(data.test_dl))


# In[143]:


dummy = predict_batch(learn.model, x)


# In[144]:


probs = F.softmax(dummy, -1)


# In[145]:


x,preds = to_np(x),to_np(probs)


# In[146]:


y = to_np(y)


# In[147]:


accuracy_np(preds,y)


# In[148]:


preds = np.argmax(preds, -1)


# In[149]:


(preds==y)


# In[150]:


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


# Check with all test data set

# In[151]:


log_preds,_ = learn.TTA(n_aug=1, is_test=True)
probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs,axis=1)
targs = data.test_ds.y
targs = to_np(targs)


# In[152]:


(preds==targs).mean()


# In[153]:


accuracy_np(probs, targs)


# ## Improve

# In[154]:


learn.fit(0.05, 5, cycle_len=1, cycle_mult=1)


# In[155]:


learn.sched.plot_lr()


# In[156]:


a = []
for i in range(5):
    log_preds,_ = learn.TTA(n_aug=1, is_test=True)
    probs = np.mean(np.exp(log_preds),0)
    targs = data.test_ds.y
    targs = to_np(targs)
    a.append(accuracy_np(probs, targs))


# In[157]:


a = np.asarray(a)
(a.mean(), a.std())


# In[34]:


preds = np.argmax(probs,axis=1)


# In[35]:


np.where(preds!=targs)


# In[36]:


data.classes[preds[999]], data.classes[targs[999]]


# In[37]:


import glob


# In[38]:


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


# In[39]:


fig, axes = plt.subplots(4, 2, figsize=(12, 8))
impred = getSamples('27691', 'train')
imtarg = getSamples('27080', 'train')
for i,ax in enumerate(axes.flat):
    if (i%2)==0:
        ima=impred[i]
    else:
        ima=imtarg[i]
    b = data.classes[preds[i]]
    c = data.classes[y[i]]
    if b!=c:
        b = b + '/'+c
    ax = show_img(ima, ax=ax)
#     draw_text(ax, (0,0), b)
    
plt.tight_layout()


# ## Mudando o domínio das Imagens Teste

# In[41]:


PATH = "data/boti/"
sz=224 #resnet restriction
arch = resnet50
bs = 16


# In[42]:


aug_tfms = [RandomRotate(90)]


# In[43]:


tfms = tfms_from_model(arch, 
                       sz=sz,
                       aug_tfms=aug_tfms,
                       crop_type=CropType.NO
                       )


# In[44]:


data = ImageClassifierData.from_paths(path=PATH, bs=bs,tfms=tfms, test_name='mags_test', test_with_labels=True)


# ### Treinando Modelo

# In[45]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# In[46]:


learn.lr_find(start_lr=1e-3, end_lr=1e2)
learn.sched.plot(n_skip=0, n_skip_end=0)


# In[47]:


learn.fit(0.02, 8)


# In[48]:


learn.fit(0.02, 5, cycle_len=1, cycle_mult=1)
learn.sched.plot_lr()


# In[158]:


log_preds,_ = learn.TTA(n_aug=7, is_test=True)
probs = np.mean(np.exp(log_preds),0)
targs = data.test_ds.y
targs = to_np(targs)
accuracy_np(probs, targs)


# In[159]:


log_preds,_ = learn.TTA(n_aug=50, is_test=True)
probs = np.mean(np.exp(log_preds),0)
targs = data.test_ds.y
targs = to_np(targs)
accuracy_np(probs, targs)


# In[160]:


pbs = []
acc = []
for i in range(10):
    log_preds,_ = learn.TTA(n_aug=7, is_test=True)
    probs = np.mean(np.exp(log_preds),0)
    targs = data.test_ds.y
    targs = to_np(targs)
    probs_sorted = np.argsort(probs, axis=1)
    pbs.append(probs_sorted)
    preds1=probs_sorted[:,-1]
    preds2=probs_sorted[:,-2]
    _acc = ((preds1==targs) | (preds2==targs)).mean()
    print (_acc)
    acc.append(((preds1==targs).mean(), _acc))


# In[161]:


acc = np.asarray(acc)
acc[:,0].mean(), acc[:,0].std()


# In[ ]:


probs_sorted = np.argsort(probs, axis=1)
probs_sorted.shape, targs.shape
preds = np.argmax(probs, 1)
preds


# In[ ]:


preds1=probs_sorted[:,-1]
preds2=probs_sorted[:,-2]
preds3=probs_sorted[:,-3]
preds4=probs_sorted[:,-4]
preds5=probs_sorted[:,-5]


# In[ ]:


(preds1==targs).mean(), (preds2==targs).mean()


# In[ ]:


((preds1==targs) | (preds2==targs)).mean()


# In[ ]:


e = np.asarray(np.where(preds!=targs))
e, len(e[0])


# In[ ]:


e[0,5], len(e[0])


# In[ ]:


probs[e[0]].mean()


# In[ ]:


a = np.setdiff1d(np.array(range(probs.shape[0])), e)


# In[ ]:


(np.asarray(probs[a])).mean()


# In[ ]:


data.classes[preds[1]], data.classes[targs[1]]


# In[ ]:


fig, axes = plt.subplots(6, 4, figsize=(12, 15))

for i,ax in enumerate(axes.flat):
    predcat = data.classes[preds[e[0,i]]]
    targcat = data.classes[targs[e[0,i]]]
    impred = getSamples(predcat, 'train')[0]
    imtarg = getSamples(targcat, 'mags_test')[0]
    
    im = np.hstack([impred, imtarg])
    ax.set_title(predcat+'/'+targcat )
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = show_img(im, ax=ax)
    
#     draw_text(ax, (0,0), predcat+'/'+targcat)
plt.axis('off')
plt.tight_layout()
    


# In[ ]:


coffees = ['20600', '20601', '20602']
florattas = ['25472', '25475', '25477']
matches = ['70043','70045', '70047', '70049', '70055']


# In[ ]:


('/').join(coffees)


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(12, 15))

for i,ax in enumerate(axes.flat):
    im = getSamples(coffees[i], 'train')[0]
    ax.set_title(coffees[i])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = show_img(im, ax=ax)
    
#     draw_text(ax, (0,0), predcat+'/'+targcat)
plt.axis('off')
plt.tight_layout()


# In[ ]:


plt.clf()
florattas = ['25472', '25475', '25477']
fig, axes = plt.subplots(1, 3, figsize=(12, 15))

for i,ax in enumerate(axes.flat):
    im = getSamples(florattas[i], 'train')[7]
    ax.set_title(florattas[i])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = show_img(im, ax=ax)
    
#     draw_text(ax, (0,0), predcat+'/'+targcat)
plt.axis('off')
plt.tight_layout()


# In[ ]:


plt.clf()

fig, axes = plt.subplots(1, 5, figsize=(12, 15))

for i,ax in enumerate(axes.flat):
    im = getSamples(matches[i], 'train')[3]
    ax.set_title(matches[i])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = show_img(im, ax=ax)
    
#     draw_text(ax, (0,0), predcat+'/'+targcat)
plt.axis('off')
plt.tight_layout()

