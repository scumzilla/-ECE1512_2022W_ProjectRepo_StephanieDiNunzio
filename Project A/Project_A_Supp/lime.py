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
import math

np.random.seed(50)

def perturb_img_1d(img, perturb, sec_size):
    mask = np.zeros(img.shape)[0]
    not_masked = np.where(perturb == 1)[0]
    for i, val in enumerate(not_masked):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size*i + j] = 1
    perturbed_img = img*mask[:,np.newaxis]
    return perturbed_img

def perturb_img(img, perturb, segs):
    mask = np.zeros(segs.shape)
    not_masked = np.where(perturb == 1)[0]
    for i in not_masked:
        mask[segs == i] = 1
    perturbed_img = copy.deepcopy(img)
    perturbed_img = perturbed_img*mask[:,:,np.newaxis]
    return perturbed_img


def LIME_img(img, model, label, num_perturb=300, kernel_size=4,max_dist=200, ratio=0.2, kernel_w=0.25, num_feats=10):
    superpixels = skimage.segmentation.quickshift(img, kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    num_superpixels = np.unique(superpixels).shape[0]
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
    lime_model.fit(perturbs, preds[:,:,label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask = np.zeros(num_superpixels) 
    mask[top_feats]= True 
    perturbed_img = perturb_img(img,mask,superpixels)
    return perturbed_img

def LIME_1d(img, model, label, num_perturb=300, sec_size=4, kernel_w=0.25, num_feats=4):
    #sec_size = size of superpixels is chosen as an input
    #num_feats = number of superpixels to output as explanation
    
    num_superpixels = math.ceil(img.shape[1]/sec_size)
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed_img = perturb_img_1d(img, i, sec_size)
        pred = model.predict(perturbed_img)
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis,:]
    dists = sklearn.metrics.pairwise_distances(perturbs, orig_img, metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2)/kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(perturbs, preds[:,:,label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask_superpixel = np.zeros(num_superpixels) 
    mask_superpixel[top_feats]= True 
    mask = np.zeros(img.shape[1])
    for i, val in enumerate(mask_superpixel):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size*i + j] = 1
    perturbed_img = img*mask[:,np.newaxis]
    
    c_exp = np.zeros(img.shape[1])
    for i, c_i in enumerate(c):
        for j in range(sec_size):
                if c_i > 0:
                    c_exp[sec_size*i + j] = c_i

    return c_exp, perturbed_img[0] 