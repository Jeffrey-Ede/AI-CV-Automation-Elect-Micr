from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import Image

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

#tf.logging.set_verbosity(tf.logging.DEBUG)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def gen_lq(img, mean):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img, axis=None)))

    return scale0to1(lq)

model_dir = "E:/models/noise2/"
cropsize = 1024
counter = 40844

loaded_img = imread("E:/stills_hq/train/train190.tif", mode='F')
loaded_img = scale0to1(cv2.resize(loaded_img, (cropsize, cropsize), interpolation=cv2.INTER_AREA))
img_orig = loaded_img
loaded_img = gen_lq(loaded_img, 1)
img_orig *= np.mean(loaded_img) / np.mean(img_orig)

predict_fn = tf.contrib.predictor.from_saved_model(model_dir+"model-"+str(counter)+"/")
prediction1 = predict_fn({"lq": loaded_img})

cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
cv2.imshow("dfsd", loaded_img)
cv2.waitKey(0)

cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
cv2.imshow("dfsd", prediction1['prediction'].reshape(cropsize, cropsize))
cv2.waitKey(0)

cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
cv2.imshow("dfsd", img_orig.reshape(cropsize, cropsize))
cv2.waitKey(0)

#Save
dir = "C:/dump/"
save_noise = dir+"noise6.tif"
save_reconstruct = dir+"reconstruct6.tif"
save_orig = dir+"orig6.tif"

Image.fromarray(loaded_img).save( save_noise )
Image.fromarray(prediction1['prediction'].reshape(cropsize, cropsize)).save( save_reconstruct )
Image.fromarray(img_orig).save( save_orig )