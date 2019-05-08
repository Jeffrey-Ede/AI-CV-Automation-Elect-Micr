import numpy as np
from scipy.misc import imread
from scipy.stats import entropy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 400
fontsize = 7
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import cv2

image_locs = [
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-44/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-43/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-39/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-37/errors.tif"]
x_titles = [
    "Capped, Weighted", 
    "Capped", 
    "Weighted", 
    "Adversarial"]
image_locs = image_locs[:3]
x_titles = x_titles[:3]

def image_entropy(img, num_bins=2**8, eps=1.e-6):

    hist, _ = np.histogram(img.astype(np.float32), bins=256, range=(0, 1))
    print(hist.shape)
    entr = entropy(hist + eps, base=2) #Base 2 is Shannon entropy

    return entr

images = [imread(loc, mode="F") for loc in image_locs]
imgs = []
for img in images:

    norm_img = 0.15*(img - np.mean(img))/np.std(img)
    norm_img += 0.5

    imgs.append(norm_img)

    print(image_entropy(norm_img))




def scale0to1(img):
    
    min = np.min(img)
    max = np.max(img)

    print(min, max)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

#Width as measured in inkscape
scale = 2
width = scale * 2.2
height = scale* (width / 1.618) / 2.2

print(np.mean(imgs[0]), np.mean(imgs[1]))

#Image.fromarray(imgs[1]).save('general_abs_err.tif')

set_mins = []
set_maxs = []

for img in imgs:
    set_mins.append(0)
    set_maxs.append(1)

w = h = 512

subplot_cropsize = 64
subplot_prop_of_size = 0.6
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.2
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

print(imgs[1])

f=plt.figure(figsize=(1, 4))
columns = 3
rows = 1

for i in range(1):
    for j in range(1, columns+1):
        img = np.ones(shape=(side,side))
        img[:w, :w] = scale0to1(imgs[columns*i+j-1])
        img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
                                                                    (subplot_side, subplot_side), 
                                                                    cv2.INTER_CUBIC)
        img = img.clip(0., 1.)
        k = i*columns+j
        ax = f.add_subplot(rows, columns, k)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

        ax.set_frame_on(False)
        if not i:
            ax.set_title(x_titles[j-1])#, fontsize=fontsize)

f.subplots_adjust(wspace=0.0, hspace=0.0)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

f.set_size_inches(width, height)

#plt.show()

f.savefig('systematic_errors.pdf', bbox_inches='tight')
