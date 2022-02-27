from winreg import HKEY_LOCAL_MACHINE
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

np.random.seed(50)

def perturb_img(img, perturb, segs):
    mask = np.zeros(segs.shape)
    not_masked = np.where(perturb == 1)[0]
    for i in not_masked:
        mask[segs == i] = 1
    perturbed_img = copy.deepcopy(img)
    perturbed_img = perturbed_img*mask[:,:,np.newaxis]
    return perturbed_img


def LIME(img, model, label, num_perturb=300, kernel_w=0.25, num_feats=4):
    superpixels = skimage.segmentation.quickshift(img, kernel_size=4,max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]
    skimage.io.imshow(skimage.segmentation.mark_boundaries(img, superpixels))
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed_img = perturb_img(img, i, superpixels)
        pred = model.predict(perturbed_img[np.newaxis,:,:,:])
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis,:]
    dists = sklearn.metrics.pairwise_distances(perturbs, orig_img, metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2)/kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(x=perturbs, y=preds[:,:,label], sample_weight=weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask = np.zeros(num_superpixels) 
    mask[top_feats]= True 
    skimage.io.imshow(perturb_img(img,mask,superpixels) )
