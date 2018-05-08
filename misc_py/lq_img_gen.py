import time 
import numpy as np
import multiprocessing
import cv2

def gen_lq(img, mean):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    #Rescale between 0 and 1
    min = np.min(lq)
    lq = (lq-min) / (np.max(lq)-min)

    return lq

if __name__ == "__main__":

    from scipy.misc import imread
    
    path = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/2100plus_dm3/'
    file = 'reaping74.tif'
    saveLoc = 'C:/dump'
    
    img = imread(path+file, mode='F')
    cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    cv2.imshow("dfsd", img.reshape((2048,2048)))
    cv2.waitKey(0)
    #img = np.random.rand(2048*2048)
    print("hi"),
    img = gen_lq(img, 12)
    print("hi2")
    cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    cv2.imshow("dfsd", img)
    cv2.waitKey(0)
