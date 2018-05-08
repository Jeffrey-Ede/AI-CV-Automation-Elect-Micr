##Copy files from one directory to another to partition the dataset
from shutil import copyfile
import os
from random import shuffle

import time

time.sleep(8*60*60)

#Get files
inDir = "E:/stills_all/stills/"
files = os.listdir(inDir)
shuffle(files)
L = len(files)

outDir = "F:/stills_all/"

for i in range(1, int(0.7*L)):
    src = inDir+files[i]
    dst = outDir+"train/train"+str(i)+".tif"

    copyfile(src, dst)

    print(dst)

for j, i in enumerate(range(int(0.7*L)+1, int(0.85*L)), 1):
    src = inDir+files[i]
    dst = outDir+"val/val"+str(j)+".tif"

    copyfile(src, dst)

    print(dst)

for j, i in enumerate(range(int(0.785*L)+1, int(L)), 1):
    src = inDir+files[i]
    dst = outDir+"test/test"+str(j)+".tif"

    copyfile(src, dst)

    print(dst)

#counter = 1
#for filename in os.listdir(inDir):
#    src = inDir+filename
#    dst = outDir+"reaping"+str(counter)+".tif"

#    copyfile(src, dst)
#    counter += 1

#    print(counter-14693)
