import os
import numpy as np

lossDir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/autoencoder_xception-multi-gpu-1/"

addrs = os.listdir(lossDir)
addrs = [lossDir+file for file in addrs if '.npy' in file]

print([np.mean(np.load(loc)) for loc in addrs])