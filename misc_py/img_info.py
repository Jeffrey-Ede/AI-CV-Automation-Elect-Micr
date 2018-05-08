import os
import numpy as np

import math

from scipy.misc import imread
from scipy.signal import convolve2d

def estimate_noise(I):
    '''Estimate image noise'''

    H, W = I.shape

    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

path = '//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/stills/'
list = []
for i, file in enumerate(os.listdir(path)):
    
    img = imread(path+file, mode='F')
    
    mean = np.mean(img)

    print(i, mean)#, estimate_noise(img))

    list.append(mean)

count1000 = 0
count2000 = 0
count3000 = 0
count5000 = 0

for mean in list:
    if mean >= 1000:
        count1000 += 1
    if mean >= 2000:
        count2000 += 1
    if mean >= 3000:
        count3000 += 1
    if mean >= 5000:
        count5000 += 1


print(count1000, count2000, count3000, count5000)