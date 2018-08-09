from apply_autoencoders import Micrograph_Autoencoder

from scipy.misc import imread
from PIL import Image
import os
import numpy as np

cropsize = 20

ckpt_loc = 'G:/noise-removal-kernels-TEM/autoencoder/16/model/'
nn = Micrograph_Autoencoder(checkpoint_loc=ckpt_loc,
                            visible_cuda='1',
                            encoding_features=16)

data_loc = "G:/unaltered_TEM_crops-171x171/"
save_loc0 = "G:/noise-removal-kernels-TEM/data/orig/"
save_loc = "G:/noise-removal-kernels-TEM/data/16/"

files = os.listdir(data_loc)
num_files = len(files)
print("Num files: {}".format(num_files))

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

counter = 1
for k, file in enumerate(files):
    print("Train file {} of {}".format(k, num_files))

    try:
        img = imread(data_loc+file, mode="F")
        img = img[:160, :160]
        nn_img = nn.denoise_crop(img)

        c = np.min(img)
        m = np.mean(img)-c
        img = (img - c) / m

        c = np.min(nn_img)
        m = np.mean(nn_img)-c
        nn_img = (nn_img - c) / m

        if img.shape[0] >= cropsize and img.shape[1] >= cropsize:
            
            #for i in range(0, img.shape[0]-cropsize+1, cropsize):
            #    for j in range(0, img.shape[1]-cropsize+1, cropsize):

            i = np.random.randint(20, 160-20-20)
            j = np.random.randint(20, 160-20-20)

            Image.fromarray(nn_img[i:(i+cropsize), j:(j+cropsize)]).save( save_loc+str(counter)+".tif" )
            Image.fromarray(img[i:(i+cropsize), j:(j+cropsize)]).save( save_loc0+str(counter)+".tif" )
            counter += 1
    except:
        print('error')

    if counter >= 6077:
        break