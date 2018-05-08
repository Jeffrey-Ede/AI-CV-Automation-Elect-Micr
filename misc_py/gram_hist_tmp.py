import numpy as np
import os
from scipy.misc import imread
from random import shuffle

in_dir = "E:/stills_hq/train/"
#in_dir = "E:/stills_hq/val/"
out_dir = "E:/stills_all/"
files = os.listdir(in_dir)
shuffle(files)

limit = 100000
num_comp=2

kernel = lambda x, y: x*y

#compute Gram matrix
def gram(x):
    pt_sq_norms = (x**2).sum(axis=1)
    dists_sq = np.dot(x, x.T)
    dists_sq *= -2
    dists_sq += pt_sq_norms.reshape(-1, 1)
    dists_sq += pt_sq_norms
    return dists_sq

skips = 0
gram_ssds = []
print(len(files))
loop_num = 0
not_break = True
while not_break:
    for i in range(len(files)):
        print("Iter: {}".format(i))

        j = i+1+loop_num
        j -= (j//len(files))*len(files)

        src1 = in_dir+files[i]
        src2 = in_dir+files[j]

        img1 = imread(src1, mode='F')
        if np.sum(np.isfinite(img1)) != 2048*2048:
            skips += 1
            continue

        img2 = imread(src2, mode='F')
        if np.sum(np.isfinite(img2)) != 2048*2048:
            skips += 1
            continue
    
        gram_ssds.append( np.mean( (gram(img1)-gram(img2))**2 ) )

        if i*num_comp > limit:
            not_break = False
            break
    loop_num += 1

np.save(out_dir+"diffs_of_gram_matrices_hq.npy", np.asarray(gram_ssds))

