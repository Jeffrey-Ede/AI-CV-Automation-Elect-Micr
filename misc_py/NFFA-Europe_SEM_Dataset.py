import numpy as np
import os
import cv2
from PIL import Image

parent = "Z:/Jeffrey-Ede/NFFA-EUROPE_SEM_Dataset/"

parent32 = "Z:/Jeffrey-Ede/NFFA-EUROPE_SEM_Dataset/32x32/"
parent64 = "Z:/Jeffrey-Ede/NFFA-EUROPE_SEM_Dataset/64x64/"
parent96 = "Z:/Jeffrey-Ede/NFFA-EUROPE_SEM_Dataset/96x96/"

subs = ["Biological",
        "Fibres",
        "Films_Coated_Surface",
        "MEMS_devices_and_electrodes",
        "Nanowires",
        "Particles",
        "Patterned_surface",
        "Porous_Sponge",
        "Powder",
        "Tips"]

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

#Load each file in NFFA-EUROPE SEM dataset and save as 32x32 and 96x96
#after cropping the information boxes
total_files = 0
num_subs = len(subs)
for i, sub in enumerate(subs):
    daughter = sub+"/"+sub+"/"
    dir = parent+daughter
    files = os.listdir(dir)

    dir32 = parent32+daughter
    dir64 = parent64+daughter
    dir96 = parent96+daughter

    num_files = len(files)
    total_files += num_files
    for j, file in enumerate(files):
        try:
            print("Subset {} of {}. File {} of {}.".format(i+1, num_subs, j+1, num_files))

            full_file = dir+file

            img = np.asarray(Image.open(full_file))
            img = img[:,:,0]

            #Find start of box
            means = list(np.mean(img, axis=1))
            indices = [i for i in range(len(means))]
            sorted_indices = [x for _,x in sorted(zip(means, indices))]
            box_start_idx = min(sorted_indices[:2])

            #Crop sqare from top-left that does not contain box
            img = img[:box_start_idx,:].astype(np.float32)

            def to_datasets(img, label):
                #Save 32x32 and 96x96 variants
                img32 = cv2.resize(img, (32,32), cv2.INTER_AREA)
                img64 = cv2.resize(img, (64,64), cv2.INTER_AREA)
                img96 = cv2.resize(img, (96,96), cv2.INTER_AREA)

                new_filename = file.replace(".jpg", "_"+label+".tif")
                Image.fromarray( img32.astype(np.uint8) ).save( dir32+new_filename )
                Image.fromarray( img64.astype(np.uint8) ).save( dir64+new_filename )
                Image.fromarray( img96.astype(np.uint8) ).save( dir96+new_filename )
                return

            to_datasets(img[:,:img.shape[0]], "left")
            to_datasets(img[:,((img.shape[1]-img.shape[0])//2):
                            ((img.shape[1]-img.shape[0])//2+img.shape[0])], "middle")
            to_datasets(img[:,(img.shape[1]-img.shape[0]-1):], "right")
        except:
            continue


print(total_files)